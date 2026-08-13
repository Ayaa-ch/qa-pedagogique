"""
Pipeline de correction — correction.py :
1. LANGUE : si n'importe quel fichier du lot a >= 30% de blocs hors langue attendue,
   le lot entier est rejeté. On ne génère jamais de corrected json, uniquement le rapport.
2. RESOLUTION (parser) : paires exercice<->correction via id puis
   titre ; rejet des references non resolues, des exercices sans correction
   et des corrections sans exercice.
3. DETECTION (coherence) : doublons d'exercices -> rejet de
   l'exercice et de sa correction ; titre global <-> exercice restant -> rejet
   ex + corr ; titre global <-> slide de resume -> rejet de la slide ; titre de
   slide de resume <-> contenu -> generation du titre.
4. SEQUENCE : renumérotation des exercices par ordre physique, réordonnancement
   physique des corrections selon l'ordre physique de leurs exercices associés,
   et alignement des titres.
 5. LABELS + VALIDATOR : NEEDS_GENERATION (marqueur manquant ->
    generation LLM au meme gabarit) ou NEEDS_RESTRUCTURE (correction globale
    complete mais incoherente -> eclatement LLM par sous-marqueur).
6. SEQUENCE DES LABELS : correction SEQUENCE_INCOHERENCE (traitée à l'étape 5).
7. RENDU : verification/correction LaTeX, HTML, Markdown.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from parser import (
    Document,
    Finding,
    Section,
    Slide,
    Block,
    BlockData,
    parse_exercise_title,
    build_exercise_index,
    resolve_exercise_reference,
    check_exercise_coverage,
    _validate_exercise_sequence,
    _validate_activities,
    _validate_slides_non_empty,
    _validate_out_of_scope_blocks,
    _check_no_exercise_sections,
    _parse_document,
)
from render import RenderFinding, check_render, _check_latex_environments
from coherence import diagnose_document, CoherenceFinding, topical_match
from generation import (
    generate_slide_title,
    generate_missing_correction_blocks,
    restructure_corrections,
)
from validator import ValidationResult, validate, _prepare_inputs
from label_sequence import (
    _extract_markers,
    _classify_marker,
    _canonical_marker,
    latin_to_abjad,
)
from language import LanguageCheckResult, check_document_language, _expected_lang

# RESOLUTION / PAIRES (parser) 
def build_pairs(sections: list[Section]) -> list[tuple[Slide, Slide]]:
    ex_slides_by_id, ex_num_to_id = build_exercise_index(sections)

    pairs: list[tuple[Slide, Slide]] = []
    for sec in sections:
        if sec.kind != "activities":
            continue
        for slide in sec.slides:
            parsed = parse_exercise_title(slide.title)
            resolution = resolve_exercise_reference(slide, sec.id, parsed, ex_slides_by_id, ex_num_to_id)
            if resolution.status == "invalid" or resolution.exercise_id is None:
                continue

            if slide.exercise_id != resolution.exercise_id:
                slide.exercise_id = resolution.exercise_id

            exercise_slide = ex_slides_by_id.get(resolution.exercise_id)
            if exercise_slide is not None:
                pairs.append((exercise_slide, slide))

    return pairs

def _as_int_id(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, Slide):
        return value.id
    return int(value)

def apply_rejections(
    sections: list[Section],
    rejected_pairs: list[tuple[Optional[int], Optional[int]]],
) -> None:
    rejected_exercise_ids = {eid for eid, _ in rejected_pairs if eid is not None}
    rejected_correction_ids = {cid for _, cid in rejected_pairs if cid is not None}

    for sec in sections:
        if sec.kind == "exercises":
            sec.slides = [s for s in sec.slides if s.id not in rejected_exercise_ids]
        elif sec.kind == "activities":
            sec.slides = [s for s in sec.slides if s.id not in rejected_correction_ids]

def filter_pairs(
    pairs: list[tuple[Slide, Slide]],
    rejected_pairs: list[tuple[Optional[int], Optional[int]]],
) -> list[tuple[Slide, Slide]]:
    rejected_exercise_ids = {eid for eid, _ in rejected_pairs if eid is not None}
    rejected_correction_ids = {cid for _, cid in rejected_pairs if cid is not None}
    return [
        (ex, corr) for ex, corr in pairs
        if ex.id not in rejected_exercise_ids
        and corr.id not in rejected_correction_ids
    ]

#  3. DOUBLONS + COHERENCE (coherence)
def apply_resume_rejections(sections: list[Section], rejected_slide_ids: list[int]) -> None:
    if not rejected_slide_ids:
        return
    rejected = set(rejected_slide_ids)
    for sec in sections:
        if sec.kind == "resume":
            sec.slides = [s for s in sec.slides if s.id not in rejected]

def _generate_resume_slide_titles(
    doc: Document,
    detections: list[CoherenceFinding],
) -> tuple[dict[int, str], list[CoherenceFinding]]:
    title_findings = [
        f for f in detections
        if f.kind == "slide_title" and f.verdict == "needs_generation"
    ]
    slide_lookup = {
        slide.id: slide
        for sec in doc.content.sections if sec.kind == "resume"
        for slide in sec.slides
    }
    titles: dict[int, str] = {}
    skipped: list[CoherenceFinding] = []

    for f in title_findings:
        slide = slide_lookup.get(f.slide_id)
        if slide is None:
            continue
        blocks = [b.data.text for b in slide.blocks if b.type == "text" and b.data.text]
        if not blocks:
            continue
        full_content = " ".join(blocks)
        try:
            candidate = generate_slide_title(
                doc.title, blocks, expected_lang=_expected_lang(doc.topic)
            )["title"]
        except Exception:
            skipped.append(CoherenceFinding(
                "slide_title", "review", doc.title, slide_id=f.slide_id,
                detail="generation de titre impossible — verification manuelle requise",
            ))
            continue
        if not topical_match(candidate, full_content):
            skipped.append(CoherenceFinding(
                "slide_title", "review", doc.title, slide_id=f.slide_id,
                detail=f"titre genere juge non coherent : '{candidate}' — verification manuelle requise",
            ))
        else:
            titles[f.slide_id] = candidate

    return titles, skipped

def _apply_slide_titles(doc: Document, titles: dict[int, str]) -> None:
    all_slides = {
        slide.id: slide
        for sec in doc.content.sections for slide in sec.slides
    }
    for slide_id, title in titles.items():
        slide = all_slides.get(slide_id)
        if slide is not None:
            slide.title = title

# SEQUENCE / ORDONNANCEMENT 
_EXERCISE_LABEL = {"fr": "Exercice", "ar": "تمرين"}
_CORRECTION_LABEL = {"fr": "Correction - Exercice", "ar": "تمرين"}

def reorder_and_finalize_titles(
    sections: list[Section],
    pairs: list[tuple[Slide, Slide]],
    lang: str = "fr",
) -> None:
    ex_label = _EXERCISE_LABEL.get(lang, _EXERCISE_LABEL["fr"])
    corr_label = _CORRECTION_LABEL.get(lang, _CORRECTION_LABEL["fr"])
    exercise_number: dict[int, int] = {}
    exercise_order: list[int] = []
    for sec in sections:
        if sec.kind == "exercises":
            for position, slide in enumerate(sec.slides, start=1):
                slide.title = f"{ex_label} {position}"
                exercise_number[slide.id] = position
                exercise_order.append(slide.id)

    ex_to_corr: dict[int, Slide] = {}
    for exercise_slide, correction_slide in pairs:
        ex_to_corr[exercise_slide.id] = correction_slide
        number = exercise_number.get(exercise_slide.id)
        if number is not None:
            correction_slide.title = f"{corr_label} {number}"

    for sec in sections:
        if sec.kind == "activities":
            sorted_slides = []
            for ex_id in exercise_order:
                if ex_id in ex_to_corr:
                    sorted_slides.append(ex_to_corr[ex_id])
            sec.slides = sorted_slides

# LABELS + VALIDATOR + SEQUENCE DES LABELS 
def _canonical_label(label: Optional[str]) -> Optional[str]:
    if not label:
         return None
    key = _classify_marker(label)
    return _canonical_marker(key) if key else None

def _normalize_dcc_marker(marker: Optional[str]) -> Optional[str]:
    """Normalise un marqueur DCC en forme canonique avec '.' et ')' :
    '1.a)' ou 'roman.1.a)' (ex: '1.a)', 'I.1.a)', '1)', 'a)')."""
    if not marker:
        return None
    key = _classify_marker(marker)
    if key is None:
        return None
    return f"{_canonical_marker(key)})"

def _marker_label_pairs(prompt_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for marker in _extract_markers(prompt_text):
        key = _classify_marker(marker)
        canon = _canonical_marker(key) if key else marker
        pairs.append((marker, canon))
    return pairs

def _short(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit] + "…" if len(text) > limit else text

def _normalize_dcc_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "<br><br>", text)
    text = re.sub(r"\n", "<br>", text)
    return text

def _correction_blocks_with_advice(slide: Slide) -> dict[str, tuple[Block, Optional[Block]]]:
    out: dict[str, tuple[Block, Optional[Block]]] = {}
    blocks = slide.blocks
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.type != "correction":
            i += 1
            continue
        canon = _canonical_label(block.data.label)
        if canon is None:
            i += 1
            continue
        advice = (
            blocks[i + 1]
            if i + 1 < len(blocks) and blocks[i + 1].type == "pedagogical_advice"
            else None
        )
        out[canon] = (block, advice)
        i += 2 if advice is not None else 1
    return out

def _rebuild_correction_slide_in_order(
    correction_slide: Slide,
    prompt_text: str,
    new_text_by_canon: dict[str, str],
    next_id: Iterator[int],
) -> dict[str, int]:
    """Reconstruit les blocs correction dans l'ordre des marqueurs de l'enonce.
    Retourne {label canonique: id du bloc correction cree}."""
    existing = _correction_blocks_with_advice(correction_slide)
    rebuilt: list[Block] = []
    new_block_ids: dict[str, int] = {}

    for marker, canon in _marker_label_pairs(prompt_text):
        if canon in new_text_by_canon:
            text = _normalize_dcc_text(new_text_by_canon[canon])
            block_id = next(next_id)
            new_block_ids[canon] = block_id
            rebuilt.append(Block(
                id=block_id,
                type="correction",
                data=BlockData(label=canon, text=text),
            ))
        elif canon in existing:
            corr, advice = existing[canon]
            rebuilt.append(corr)
            if advice is not None:
                rebuilt.append(advice)

    correction_slide.blocks = rebuilt
    return new_block_ids

def _strip_stale_correction_blocks(
    correction_slide: Slide,
    target_labels: set[str],
) -> int:
    if not target_labels:
        return 0

    blocks = correction_slide.blocks
    to_remove: set[int] = set()

    for i, block in enumerate(blocks):
        if block.type == "correction" and _canonical_label(block.data.label) in target_labels:
            to_remove.add(i)
            if i + 1 < len(blocks) and blocks[i + 1].type == "pedagogical_advice":
                to_remove.add(i + 1)

    removed_count = sum(1 for i in to_remove if blocks[i].type == "correction")
    correction_slide.blocks = [b for idx, b in enumerate(blocks) if idx not in to_remove]
    return removed_count

def _fix_need_generation(
    exercise_slide: Slide,
    correction_slide: Slide,
    prompt_text: str,
    questions_by_marker: dict[str, str],
    existing_blocks: list[dict],
    vres,
    expected_lang: str,
    next_id: Iterator[int],
    result: "CorrectionResult",
) -> None:
    pair_label = f"Exercice {vres.exercise_id} / Correction {vres.correction_id}"
    missing_entries = vres.need_generation

    if not missing_entries:
        return

    exercise_text_content = "\n\n".join(
        [b.data.text for b in exercise_slide.blocks if b.type == "text" and b.data.text]
        + [b.data.prompt for b in exercise_slide.blocks if b.type == "question" and b.data.prompt]
    )

    generated = generate_missing_correction_blocks(
        missing_entries, questions_by_marker, existing_blocks, exercise_text_content,
        expected_lang=expected_lang,
    )
    if not generated:
        result.validator_manual_review.append(
            f"{pair_label}: reponses LLM illisibles/aucune reponse generee "
            f"pour les marqueurs manquants — verification manuelle requise"
        )
        return

    target_canons: set[str] = set()
    for entry in missing_entries:
        canon = _canonical_label(entry.get("marqueur"))
        if canon:
            target_canons.add(canon)
    stale_count = _strip_stale_correction_blocks(correction_slide, target_canons)

    new_text_by_canon: dict[str, str] = {}
    generated_parts: list[dict] = []
    for item in generated:
        if not item.get("text"):
            continue
        label = item.get("label")
        canon = _canonical_label(label)
        if canon is None:
            continue
        text = _normalize_dcc_text(item["text"])
        new_text_by_canon[canon] = text
        generated_parts.append({
            "label": label,
            "canon": canon,
            "question": questions_by_marker.get(label, ""),
            "text": text,
        })

    new_block_ids: dict[str, int] = {}
    if new_text_by_canon:
        new_block_ids = _rebuild_correction_slide_in_order(
            correction_slide,
            prompt_text,
            new_text_by_canon,
            next_id,
        )

    details: list[str] = []
    for part in generated_parts:
        block_note = (
            f" (bloc {new_block_ids[part['canon']]})"
            if part["canon"] in new_block_ids else ""
        )
        details.append(
            f"- label {part['label']}{block_note} : "
            f"question: {_short(part['question'], 90)} "
            f"— reponse generee: {_short(part['text'], 140)}"
        )

    stale_note = ""
    if stale_count:
        stale_note = (
            f" ({stale_count} ancien(s) bloc(s) correction juge(s) "
            f"incorrect(s)/hors-sujet retire(s), advice supprime)"
        )
    message = (
        f"{pair_label}: {len(details)} reponse(s) generee(s) "
        f"pour {len(missing_entries)} marqueur(s) manquant(s){stale_note}"
    )
    if details:
        message += "\n" + "\n".join(details)
    result.content_generated.append(message)

def _fix_need_restructure(
    correction_slide: Slide,
    prompt_text: str,
    questions_by_marker: dict[str, str],
    correction_text_by_label: dict[str, str],
    vres,
    next_id: Iterator[int],
    result: "CorrectionResult",
) -> None:
    pair_label = f"Exercice {vres.exercise_id} / Correction {vres.correction_id}"
    new_text_by_canon: dict[str, str] = {}
    stale_canons: set[str] = set()
    fixed_entries: int = 0

    fix_records: list[dict] = []
    for entry in vres.need_restructure:
        markers_a_traiter = entry.get("markers_a_traiter") or []
        if not markers_a_traiter:
            continue
        global_text = correction_text_by_label.get(entry.get("correction_label"), "")
        attribution = restructure_corrections(
            markers_a_traiter, questions_by_marker, global_text
        )
        if attribution is None:
            result.validator_manual_review.append(
                f"{pair_label}: restructuration non automatisable pour "
                f"'{entry.get('correction_label', '?')}' — verification manuelle requise"
            )
            continue
        split_parts: list[dict] = []
        for marker, text in attribution.items():
            canon = _canonical_label(marker)
            if canon and text:
                new_text_by_canon[canon] = _normalize_dcc_text(text)
                split_parts.append({
                    "canon": canon,
                    "marker": marker,
                    "question": questions_by_marker.get(marker, ""),
                    "text": text,
                })
        if split_parts:
            fix_records.append({
                "correction_label": entry.get("correction_label", "?"),
                "parts": split_parts,
            })
        entry_canon = _canonical_label(entry.get("correction_label"))
        if entry_canon:
            stale_canons.add(entry_canon)
        fixed_entries += 1

    if not fixed_entries:
        return

    if stale_canons:
        _strip_stale_correction_blocks(correction_slide, stale_canons)
    new_block_ids: dict[str, int] = {}
    if new_text_by_canon:
        new_block_ids = _rebuild_correction_slide_in_order(
            correction_slide,
            prompt_text,
            new_text_by_canon,
            next_id,
        )

    for rec in fix_records:
        split_details: list[str] = []
        for part in rec["parts"]:
            block_note = (
                f" (bloc {new_block_ids[part['canon']]})"
                if part["canon"] in new_block_ids else ""
            )
            split_details.append(
                f"- {part['marker']}{block_note} : "
                f"question: {_short(part['question'], 90)} "
                f"— contenu attribue: {_short(part['text'], 140)}"
            )
        result.restructure_fixed.append(
            f"{pair_label}: '{rec['correction_label']}' divisee en "
            f"{len(split_details)} reponse(s) :\n" + "\n".join(split_details)
        )

def _fix_one_pair(
    exercise_slide: Slide,
    correction_slide: Slide,
    vres,
    expected_lang: str,
    next_id: Iterator[int],
    result: "CorrectionResult",
) -> None:
    """Applique les corrections decidees par le validateur (par labels)."""
    prompt_text = "\n\n".join(
        b.data.prompt for b in exercise_slide.blocks
        if b.type == "question" and b.data.prompt
    )

    _, questions_by_marker, correction_labels, correction_texts = _prepare_inputs(
        exercise_slide, correction_slide
    )
    existing_blocks = [
        {"label": label, "text": text}
        for label, text in zip(correction_labels, correction_texts)
    ]
    correction_text_by_label = dict(zip(correction_labels, correction_texts))

    if vres.need_generation:
        _fix_need_generation(
            exercise_slide, correction_slide, prompt_text, questions_by_marker,
            existing_blocks, vres, expected_lang, next_id, result,
        )

    if vres.need_restructure:
        _fix_need_restructure(
            correction_slide, prompt_text, questions_by_marker,
            correction_text_by_label, vres, next_id, result,
        )

    if not vres.need_generation and not vres.need_restructure and vres.content_detail:
        result.validator_manual_review.append(
            f"Exercice {vres.exercise_id} / Correction {vres.correction_id}: "
            f"{vres.content_detail} — verification manuelle requise"
        )

# ── 7. RENDU (LaTeX / HTML / Markdown) ───────────────────────────────────

def _close_braces(segment: str) -> str:
    depth = 0
    for ch in segment:
        if ch == "{":
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
    if depth == 0:
        return segment
    stripped = segment.rstrip()
    trailing_ws = segment[len(stripped):]
    return stripped + "}" * depth + trailing_ws

_MATH_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"\$(.*?)\$", re.DOTALL)

def _fix_newlines_in_math(text: str) -> str:
    def _strip_newlines(match: re.Match) -> str:
        inner = re.sub(r"\s*[\r\n]+\s*", " ", match.group(1)).strip()
        return "$$" + inner + "$$"
    return _MATH_BLOCK_RE.sub(_strip_newlines, text)

def _fix_latex_environments(text: str) -> str:
    unclosed_envs = _check_latex_environments(text)
    if not unclosed_envs:
        return text

    stripped = text.rstrip()
    trailing_ws = text[len(stripped):]
    closing = "".join(f"\\end{{{env}}}" for env in unclosed_envs)

    if stripped.endswith("$$"):
         cut = len(stripped) - 2
         return stripped[:cut] + closing + stripped[cut:] + trailing_ws

    return stripped + closing + trailing_ws

def _fix_latex(text: str) -> str:
    text = _fix_newlines_in_math(text)
    text = _fix_latex_environments(text)

    blocks: list[str] = []

    def _stash_block(match: re.Match) -> str:
        blocks.append("$$" + _close_braces(match.group(1)) + "$$")
        return f"\0{len(blocks) - 1}\0"

    text = _MATH_BLOCK_RE.sub(_stash_block, text)
    text = _MATH_INLINE_RE.sub(lambda m: "$" + _close_braces(m.group(1)) + "$", text)

    for i, block in enumerate(blocks):
        text = text.replace(f"\0{i}\0", block)

    if text.count("$$") % 2 != 0:
        text = _close_braces(text) + "$$"

    inline_text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    if inline_text.count("$") % 2 != 0:
        last_dollar_idx = inline_text.rfind("$")

        if last_dollar_idx != -1:
            before_dollar = inline_text[:last_dollar_idx]
            after_dollar  = inline_text[last_dollar_idx + 1:]

            match_word = re.search(r"^(\s*\S+)", after_dollar)
            if match_word:
                word_end     = match_word.end()
                target_token = after_dollar[:word_end]
                rest_of_text = after_dollar[word_end:]
                fixed_inline = (
                    before_dollar + "$" + _close_braces(target_token) + "$" + rest_of_text
                )
            else:
                fixed_inline = _close_braces(inline_text) + "$"

            parts = re.split(r"(\$\$.*?\$\$)", text, flags=re.DOTALL)
            fixed_parts      = []
            inline_remaining = fixed_inline
            for part in parts:
                if re.match(r"\$\$.*\$\$", part, re.DOTALL):
                    fixed_parts.append(part)
                else:
                    orig_len = len(part)
                    fixed_parts.append(inline_remaining[:orig_len])
                    inline_remaining = inline_remaining[orig_len:]
            text = "".join(fixed_parts)

    return text

def _fix_html(text: str) -> str:
    text = re.sub(r"^(?:<br\s*/?>\s*)+", "", text)
    text = re.sub(r"(?:\s*<br\s*/?>)+$", "", text)
    return re.sub(r"<(?!/?br\b)[a-zA-Z][^>]*>", "", text)

def _fix_markdown(text: str) -> str:
    if "$$" not in text:
        if text.count("**") % 2 != 0:
            text += "**"
        return text

    TOKEN_RE = re.compile(r"(\*\*|\$\$.*?\$\$)", re.DOTALL)
    raw_tokens = TOKEN_RE.split(text)

    out = []
    open_bold = False
    skip_next_close = 0

    for tok in raw_tokens:
        if tok == "**":
            if skip_next_close > 0 and not open_bold:
                skip_next_close -= 1
            else:
                open_bold = not open_bold
                out.append(tok)

        elif tok.startswith("$$"):
            if open_bold:
                if out and not out[-1].startswith("$$"):
                    out[-1] = out[-1].rstrip()
                out.append("**")
                open_bold = False
                skip_next_close += 1
            if out and out[-1] and out[-1][-1] not in ("\n", "\r"):
                out[-1] = out[-1].rstrip(" ")
                if out[-1] and out[-1][-1] not in ("\n", "\r"):
                    out.append("\n")
            out.append(tok)

        else:
            out.append(tok)

    if open_bold:
        out.append("**")

    return "".join(out)

_STRAY_NEWLINE_RE = re.compile(r"\\n(?![a-zA-Z])")

def _fix_stray_newlines(text: str) -> str:
    return _STRAY_NEWLINE_RE.sub(" ", text)

def _fix_render_text(text: str) -> str:
    return _fix_markdown(_fix_html(_fix_latex(_fix_stray_newlines(text))))

_RENDER_TEXT_FIELDS = ("text", "prompt", "method", "error", "tip")
_BLOCK_EXPORT_FIELDS = ("text", "prompt", "label", "method", "error", "tip", "variant")

def apply_render_fixes(documents: list[Document]) -> None:
    for doc in documents:
        for sec in doc.content.sections:
            for slide in sec.slides:
                for block in slide.blocks:
                    for field_name in _RENDER_TEXT_FIELDS:
                        value = getattr(block.data, field_name)
                        if value:
                            setattr(block.data, field_name, _fix_render_text(value))

# ── Orchestration ────────────────────────────────────────────────────────

def _collect_verification_findings(sections: list[Section]) -> list[Finding]:
    findings: list[Finding] = []
    for sec in sections:
        if sec.kind == "exercises":
            findings.extend(_validate_exercise_sequence(sec.slides))
    findings.extend(_validate_activities(sections))
    return findings

def _max_existing_id(node: object) -> int:
    if isinstance(node, dict):
        current = node.get("id") if isinstance(node.get("id"), int) else -1
        return max([current] + [_max_existing_id(v) for v in node.values()])
    if isinstance(node, list):
        return max([-1] + [_max_existing_id(item) for item in node])
    return -1

@dataclass
class CorrectionResult:
    documents: list[Document]
    rejected_pairs: list[tuple[Optional[int], Optional[int]]] = field(default_factory=list)
    coverage_findings: list[Finding] = field(default_factory=list)
    coherence_findings: list[CoherenceFinding] = field(default_factory=list)
    language_rejections: list[LanguageCheckResult] = field(default_factory=list)
    language_warnings: list[tuple[int, LanguageCheckResult]] = field(default_factory=list)
    resolved_findings: list[Finding] = field(default_factory=list)
    verification_findings: list[Finding] = field(default_factory=list)
    render_resolved_findings: list[RenderFinding] = field(default_factory=list)
    render_verification_findings: list[RenderFinding] = field(default_factory=list)
    generation_findings: list[CoherenceFinding] = field(default_factory=list)
    rejected_resume_slide_ids: list[int] = field(default_factory=list)
    resume_findings: list[CoherenceFinding] = field(default_factory=list)
    restructure_fixed: list[str] = field(default_factory=list)
    content_generated: list[str] = field(default_factory=list)
    validator_manual_review: list[str] = field(default_factory=list)
    parsing_rejected: list[str] = field(default_factory=list)
    structure_findings: list[Finding] = field(default_factory=list)
    remaining_markers: list[str] = field(default_factory=list)

def correct(raw_documents: list[dict]) -> CorrectionResult:
    if not isinstance(raw_documents, list):
        raise TypeError(f"correct() attend une liste de documents, recu {type(raw_documents).__name__}")

    result = CorrectionResult(documents=[])
    next_id = itertools.count(_max_existing_id(raw_documents) + 1)
    total_docs = len(raw_documents)
    has_language_rejection = False
    for doc_index, raw in enumerate(raw_documents):
        parsed = _parse_document(raw, doc_index)
        if parsed.document is None:
            continue
        doc = parsed.document
        lang_check = check_document_language(doc, doc_index)
        if lang_check.rejected:
            result.language_rejections.append(lang_check)
            has_language_rejection = True
        elif lang_check.findings:
            result.language_warnings.append((doc_index, lang_check))

    if has_language_rejection:
        return result

    for doc_index, raw in enumerate(raw_documents):
        parsed = _parse_document(raw, doc_index)
        if parsed.document is None:
            problem = f"Document {doc_index + 1}/{total_docs} : echec de parsing"
            first_problem = (parsed.field_findings or parsed.pydantic_findings or [None])[0]
            if first_problem is not None:
                problem += f" — {first_problem.message}"
            result.parsing_rejected.append(problem)
            continue

        doc = parsed.document
        sections = doc.content.sections

        no_ex_finding = _check_no_exercise_sections(doc, doc_index)
        if no_ex_finding is not None:
            result.structure_findings.append(no_ex_finding)
        result.structure_findings.extend(_validate_slides_non_empty(sections))
        result.structure_findings.extend(_validate_out_of_scope_blocks(sections))

        pairs = build_pairs(sections)

        coverage = check_exercise_coverage(sections)
        orphan_pairs = list(coverage.rejected_pairs)
        if orphan_pairs:
            apply_rejections(sections, orphan_pairs)
            result.rejected_pairs.extend(orphan_pairs)
            result.coverage_findings.extend(coverage.findings)
            pairs = filter_pairs(pairs, orphan_pairs)

        corr_by_ex_id = {exercise.id: corr for exercise, corr in pairs}
        detections = diagnose_document(doc, corr_by_ex_id)

        coherence_pairs = [
            (_as_int_id(f.slide_id), _as_int_id(f.other_slide_id))
            for f in detections
            if f.kind in ("dup_exercise_pair", "off_exercise_pair") and f.verdict == "reject"
        ]
        if coherence_pairs:
            apply_rejections(sections, coherence_pairs)
            result.rejected_pairs.extend(coherence_pairs)
            result.coherence_findings.extend(
                f for f in detections
                if f.kind in ("dup_exercise_pair", "off_exercise_pair") and f.verdict == "reject"
            )
            pairs = filter_pairs(pairs, coherence_pairs)

        resume_reject_ids = [
            _as_int_id(f.slide_id) for f in detections
            if f.kind == "resume" and f.verdict == "reject" and f.slide_id is not None
        ]
        if resume_reject_ids:
            apply_resume_rejections(sections, resume_reject_ids)
            result.rejected_resume_slide_ids.extend(resume_reject_ids)
            result.resume_findings.extend(
                f for f in detections
                if f.kind == "resume" and f.verdict == "reject"
            )

        slide_titles, skipped_titles = _generate_resume_slide_titles(doc, detections)
        _apply_slide_titles(doc, slide_titles)
        result.generation_findings.extend(skipped_titles)

        for sec in sections:
            if sec.kind == "exercises":
                result.resolved_findings.extend(_validate_exercise_sequence(sec.slides))
        result.resolved_findings.extend(_validate_activities(sections))

        expected_lang = _expected_lang(doc.topic)
        reorder_and_finalize_titles(sections, pairs, expected_lang)

        validation_results = []
        for pair in pairs:
            exercise_slide, correction_slide = pair
            try:
                validation_results.append(validate(exercise_slide, correction_slide))
            except Exception as exc:
                validation_results.append(ValidationResult(
                    exercise_id=exercise_slide.id,
                    correction_id=correction_slide.id,
                    content_status="INCOMPLETE",
                    content_detail=f"erreur de validation LLM: {exc}",
                ))

        for (exercise_slide, correction_slide), vres in zip(pairs, validation_results):
            _fix_one_pair(
                exercise_slide, correction_slide, vres,
                expected_lang, next_id, result,
            )

        for exercise_slide, correction_slide in pairs:
            try:
                vres_after = validate(exercise_slide, correction_slide)
            except Exception:
                continue
            still = [
                f"{mr.marker} ({mr.reason})"
                for mr in vres_after.marker_results
                if mr.status != "OK"
            ]
            if still:
                result.remaining_markers.append(
                    f"Exercice {exercise_slide.id} / Correction {correction_slide.id}: "
                    f"marqueur(s) encore manquant(s) apres correction: {', '.join(still)}"
                )

        for _, correction_slide in pairs:
            for block in correction_slide.blocks:
                if block.type != "correction" or not block.data.label:
                    continue
                norm = _normalize_dcc_marker(block.data.label)
                if norm:
                    block.data = block.data.model_copy(update={"label": norm})

        if expected_lang == "ar":
            for _, correction_slide in pairs:
                for block in correction_slide.blocks:
                    if block.type == "correction" and block.data.label:
                        block.data = block.data.model_copy(
                            update={"label": latin_to_abjad(block.data.label)}
                        )

        errors, warnings = check_render([doc])
        result.render_resolved_findings.extend(errors + warnings)
        apply_render_fixes([doc])
        errors, warnings = check_render([doc])
        result.render_verification_findings.extend(errors + warnings)

        result.verification_findings.extend(_collect_verification_findings(sections))

        result.documents.append(doc)

    return result

# ── Raw-JSON reconciliation ──────────────────────────────────────────────

def _index_raw_slides(node, index: dict[int, tuple[list, dict]]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _index_raw_slides(value, index)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "id" in item and "title" in item:
                index[item["id"]] = (node, item)
            _index_raw_slides(item, index)

def _index_raw_blocks(node, index: dict[int, tuple[list, dict]]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _index_raw_blocks(value, index)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "id" in item and isinstance(item.get("data"), dict):
                index[item["id"]] = (node, item)
            _index_raw_blocks(item, index)

def _index_raw_sections(node, section_of_slide: dict[int, dict], current_section=None) -> None:
    if isinstance(node, dict):
        section_context = current_section
        if "kind" in node and isinstance(node.get("slides"), list):
            section_context = node
        for key, value in node.items():
            if key == "slides" and section_context is node:
                for item in value:
                    if isinstance(item, dict) and "id" in item:
                        section_of_slide[item["id"]] = section_context
                    _index_raw_sections(item, section_of_slide, section_context)
            else:
                _index_raw_sections(value, section_of_slide, section_context)
    elif isinstance(node, list):
        for item in node:
            _index_raw_sections(item, section_of_slide, current_section)

def _remove_from_parent(parent_list: list, node: dict) -> None:
    parent_list[:] = [item for item in parent_list if item is not node]

def _reorder_data_label_first(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    label = data.get("label")
    if label is None or "label" not in data:
        return data
    ordered = {"label": label}
    for key, value in data.items():
        if key != "label":
            ordered[key] = value
    return ordered

def export_corrected_and_orphans(
    raw_documents: list[dict],
    result: CorrectionResult,
) -> tuple[list[dict], list[dict]]:
    raw_index: dict[int, tuple[list, dict]] = {}
    raw_block_index: dict[int, tuple[list, dict]] = {}
    section_of_slide: dict[int, dict] = {}
    for raw in raw_documents:
        _index_raw_slides(raw, raw_index)
        _index_raw_blocks(raw, raw_block_index)
        _index_raw_sections(raw, section_of_slide)

    orphans_by_section: dict[int, dict] = {}
    for exercise_id, correction_id in result.rejected_pairs:
        slide_id = exercise_id if exercise_id is not None else correction_id
        entry = raw_index.get(slide_id)
        if entry is None:
            continue
        parent_list, node = entry

        section = section_of_slide.get(slide_id)
        if section is not None:
            bucket = orphans_by_section.setdefault(section.get("id"), {
                "id": section.get("id"),
                "title": section.get("title"),
                "kind": section.get("kind"),
                "slides": [],
            })
            bucket["slides"].append(node)

        _remove_from_parent(parent_list, node)

    for doc in result.documents:
        for sec in doc.content.sections:
            for slide in sec.slides:
                entry = raw_index.get(slide.id)
                if entry is None:
                    continue
                _, node = entry
                node["title"] = slide.title
                if "exercise_id" in node:
                    node["exercise_id"] = slide.exercise_id

                new_blocks: list[dict] = []
                for block in slide.blocks:
                    block_entry = raw_block_index.get(block.id)
                    if block_entry is not None:
                        _, block_node = block_entry
                        data_node = block_node.get("data")
                        if isinstance(data_node, dict):
                            for field_name in _BLOCK_EXPORT_FIELDS:
                                value = getattr(block.data, field_name)
                                if value is not None:
                                    data_node[field_name] = value
                        if block.type == "correction":
                            block_node["data"] = _reorder_data_label_first(block_node.get("data"))
                        new_blocks.append(block_node)
                    else:
                        block_dict = {
                            "id": block.id,
                            "type": block.type,
                            "data": {
                                field_name: getattr(block.data, field_name)
                                for field_name in (
                                    "text", "prompt", "label", "method",
                                    "error", "tip", "variant",
                                )
                                if getattr(block.data, field_name) is not None
                            },
                        }
                        if block.type == "correction":
                            block_dict["data"] = _reorder_data_label_first(block_dict["data"])
                        new_blocks.append(block_dict)
                node["blocks"] = new_blocks

    return raw_documents, list(orphans_by_section.values())

def _summarize_items(items: list[str], limit: int = 15) -> list[str]:
    """Regroupe les messages identiques et plafonne le nombre de lignes."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    grouped = [msg if n == 1 else f"{msg} (x{n})" for msg, n in counts.items()]
    if len(grouped) > limit:
        grouped = grouped[:limit] + [f"(+{len(grouped) - limit} autre(s) message(s) distinct(s))"]
    return grouped

def _summarize_render(findings, limit: int = 12) -> list[str]:
    """Classifie les erreurs de rendu par severite + message, avec ids block/slide."""
    groups: dict[tuple[str, str], list[str]] = {}
    for f in findings:
        loc = f"b{f.block_id}" if f.block_id is not None else "?"
        if f.slide_id is not None:
            loc += f"/s{f.slide_id}"
        groups.setdefault((f.severity, f.message), []).append(loc)
    lines: list[str] = []
    for (sev, msg), locs in groups.items():
        tag = "[ERROR]" if sev == "error" else "[WARNING]"
        shown = locs[:limit]
        rest = f" (+{len(locs) - limit} autres)" if len(locs) > limit else ""
        lines.append(f"{tag} x{len(locs)} — {_short(msg, 60)} — {', '.join(shown)}{rest}")
    return lines

def _build_report_steps(result: CorrectionResult) -> list[tuple[str, list[str], str]]:
    """Etapes non vides uniquement, resumees et classifiees par severite."""
    steps: list[tuple[str, list[str], str]] = []

    if result.parsing_rejected:
        steps.append(("PARSING (echec de parsing)",
                      _summarize_items(result.parsing_rejected),
                      "document(s) ecarte(s) avant correction"))

    if result.structure_findings:
        steps.append(("STRUCTURE",
                      _summarize_items([str(f) for f in result.structure_findings]),
                      f"{len(result.structure_findings)} point(s) signale(s)"))

    if result.coverage_findings:
        steps.append(("REJET ORPHELINS",
                      _summarize_items([str(f) for f in result.coverage_findings]),
                      f"{len(result.coverage_findings)} paire(s) orpheline(s) supprimee(s)"))

    if result.coherence_findings:
        steps.append(("DOUBLONS / HORS-SUJET",
                      _summarize_items([str(f) for f in result.coherence_findings]),
                      f"{len(result.coherence_findings)} paire(s) rejetee(s)"))

    if result.resume_findings:
        steps.append(("SLIDES RESUME",
                      _summarize_items([str(f) for f in result.resume_findings]),
                      f"{len(result.resume_findings)} slide(s) retiree(s)"))

    if result.generation_findings:
        steps.append(("TITRES RESUME",
                      _summarize_items([str(f) for f in result.generation_findings]),
                      "titres regeneres, cas restants en verification manuelle"))

    if result.resolved_findings:
        steps.append(("SEQUENCE",
                      _summarize_items([str(f) for f in result.resolved_findings]),
                      f"{len(result.resolved_findings)} incoherence(s) corrigee(s)"))

    validation_items = (
        result.restructure_fixed + result.content_generated + result.validator_manual_review
    )
    if validation_items:
        items = _summarize_items(validation_items, limit=15)
        steps.append(("VALIDATION",
                      items,
                      f"{len(result.restructure_fixed)} label(s) realigne(s), "
                      f"{len(result.content_generated)} reponse(s) generee(s), "
                      f"{len(result.validator_manual_review)} cas en verification manuelle"))

    if result.remaining_markers:
        steps.append(("VALIDATION APRES CORRECTION",
                      _summarize_items(result.remaining_markers),
                      f"{len(result.remaining_markers)} paire(s) avec marqueur(s) "
                      f"encore manquant(s) apres correction"))

    render_findings = result.render_resolved_findings + result.render_verification_findings
    if render_findings:
        steps.append(("RENDU (LaTeX/HTML/Markdown)",
                      _summarize_render(render_findings, limit=10),
                      f"{len(result.render_resolved_findings)} corrigee(s), "
                      f"{len(result.render_verification_findings)} restante(s)"))

    if result.verification_findings:
        steps.append(("VERIFICATION FINALE",
                      _summarize_items([str(f) for f in result.verification_findings]),
                      f"{len(result.verification_findings)} point(s) restant(s)"))

    return steps
