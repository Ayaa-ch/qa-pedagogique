from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Literal
from pydantic import BaseModel, field_validator, model_validator

# ── Pydantic models ──────────────────────────────────────────────────────

class BlockData(BaseModel):
    text:    Optional[str] = None
    prompt:  Optional[str] = None
    label:   Optional[str] = None
    method:  Optional[str] = None
    error:   Optional[str] = None
    tip:     Optional[str] = None
    variant: Optional[str] = None

    @field_validator("text", "prompt", "label", "method", "error", "tip", mode="before")
    @classmethod
    def _blank_as_missing(cls, v):
        return None if v == "" else v

class Block(BaseModel):
    id: int
    type: Literal["text", "question", "course_reminder", "correction", "pedagogical_advice"]
    data: BlockData

    @model_validator(mode="after")
    def _check_required_fields(self):
        missing = []
        if self.type == "question" and not self.data.prompt:
            missing.append("data.prompt")
        if self.type == "text" and not self.data.text:
            missing.append("data.text")
        if self.type == "correction":
            if not self.data.label:
                missing.append("data.label")
            if not self.data.text:
                missing.append("data.text")
        if missing:
            raise ValueError(f"Block {self.id} ({self.type}) missing: {missing}")
        return self

class Slide(BaseModel):
    id: int
    title: str
    order: Optional[int] = None
    exercise_id: Optional[int] = None
    blocks: list[Block]

class Section(BaseModel):
    id: int
    title: str
    kind: Literal["resume", "exercises", "activities"]
    slides: list[Slide]

class ContentMeta(BaseModel):
    language: str

class Content(BaseModel):
    schemaVersion: str
    meta: ContentMeta
    sections: list[Section]

class Document(BaseModel):
    title: str
    topic: str
    grade: list[str]
    layout_mode: Optional[str] = None
    content: Content

# ── Findings ─────────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity:   Literal["error", "warning"]
    category:   str
    message:    str
    doc_index:  Optional[int] = None
    section_id: Optional[int] = None
    slide_id:   Optional[int] = None
    block_id:   Optional[int] = None
    field_name: Optional[str] = None

    def __str__(self):
        tag = "[ERROR]" if self.severity == "error" else "[WARNING]"
        return f"{tag} {self.message}"

def _finding(severity, category, message, **kw) -> Finding:
    return Finding(severity=severity, category=category, message=message, **kw)

def _err(category, message, **kw) -> Finding:
    return _finding("error", category, message, **kw)

def _warn(category, message, **kw) -> Finding:
    return _finding("warning", category, message, **kw)

@dataclass
class CoverageResult:
    findings: list[Finding]
    rejected_pairs: list[tuple[Optional[int], Optional[int]]]

# ── Title parsing / exercise indexing ────────────────────────────────────

@dataclass
class TitleParseResult:
    status: Literal["valid", "recoverable", "invalid"]
    number: Optional[int]

_RE_VALID = re.compile(
    r"^(?:exercice|correction|تمرين)\s+\d+\b",
    re.IGNORECASE,
)
_RE_ANY_NUMBER = re.compile(r"\d+")

def parse_exercise_title(title: Optional[str]) -> TitleParseResult:
    text = (title or "").strip()
    if not text:
        return TitleParseResult("invalid", None)

    match = _RE_ANY_NUMBER.search(text)
    if not match:
        return TitleParseResult("invalid", None)

    status = "valid" if _RE_VALID.match(text) else "recoverable"
    return TitleParseResult(status, int(match.group()))

def build_exercise_index(sections: list[Section]) -> tuple[dict[int, Slide], dict[int, int]]:
    ex_slides_by_id: dict[int, Slide] = {
        s.id: s for sec in sections if sec.kind == "exercises" for s in sec.slides
    }
    ex_num_to_id: dict[int, int] = {}
    for slide_id, slide in ex_slides_by_id.items():
        number = parse_exercise_title(slide.title).number
        if number is not None and number not in ex_num_to_id:
            ex_num_to_id[number] = slide_id
    return ex_slides_by_id, ex_num_to_id


def _build_partial(model_cls, data):
    if not isinstance(data, dict):
        return data

    fields = {}
    for key, value in data.items():
        field_info = model_cls.model_fields.get(key)
        if field_info is None:
            fields[key] = value
            continue

        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            item_type = annotation.__args__[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                fields[key] = [
                    _build_partial(item_type, item) if isinstance(item, dict) else item
                    for item in (value or [])
                ]
                continue
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
            fields[key] = _build_partial(annotation, value)
            continue
        fields[key] = value

    return model_cls.model_construct(**fields)

# ── Document-level validators ────────────────────────────────────────────

def _validate_document_fields(doc: dict, doc_index: int) -> list[Finding]:
    findings = [
        _err("document_fields", f"Document champ manquant: {name}", doc_index=doc_index, field_name=name)
        for name in ("title", "topic")
        if not doc.get(name) and doc.get(name) != 0
    ]
    grade = doc.get("grade")
    if not (isinstance(grade, list) and any(isinstance(g, str) and g.strip() for g in grade)):
        findings.append(_err("document_fields", "Document champ manquant: grade",
                              doc_index=doc_index, field_name="grade"))
    return findings

def _translate_pydantic_errors(exc: Exception, doc_index: int,
                                skip_fields: frozenset[str] = frozenset()) -> list[Finding]:
    if not hasattr(exc, "errors"):
        return [_err("required_blocks", str(exc), doc_index=doc_index)]

    findings = []
    for error in exc.errors():
        loc = error["loc"]
        if len(loc) == 1 and loc[0] in skip_fields:
            continue
        path = " > ".join(str(part) for part in loc)
        findings.append(_err("required_blocks", f"[{path}]: {error['msg']}",
                              doc_index=doc_index, field_name=path))
    return findings

def _validate_slides_non_empty(sections: list[Section]) -> list[Finding]:
    findings = []
    for sec in sections:
        if not sec.slides:
            findings.append(_err("required_blocks",
                f"Section {sec.id} '{sec.title}' ({sec.kind}): 0 slides", section_id=sec.id))
            continue
        for slide in sec.slides:
            if not slide.blocks:
                findings.append(_err("required_blocks",
                    f"Slide {slide.id} '{slide.title}': slide vide (0 blocs)",
                    section_id=sec.id, slide_id=slide.id))
    return findings

def _validate_out_of_scope_blocks(sections: list[Section]) -> list[Finding]:
    findings = []
    for sec in sections:
        for slide in sec.slides:
            kw = dict(section_id=sec.id, slide_id=slide.id)
            for block in slide.blocks:
                if block.type == "course_reminder" and not block.data.text:
                    findings.append(_warn("out_of_scope",
                        f"Block {block.id} course_reminder: data.text manquant",
                        field_name="data.text", block_id=block.id, **kw))
                elif block.type == "pedagogical_advice":
                    findings += [
                        _warn("out_of_scope", f"Block {block.id} pedagogical_advice: data.{attr} manquant",
                              field_name=f"data.{attr}", block_id=block.id, **kw)
                        for attr in ("method", "error", "tip") if not getattr(block.data, attr)
                    ]
    return findings

def _check_no_exercise_sections(doc: Document, doc_index: int) -> Optional[Finding]:
    if any(sec.kind in ("exercises", "activities") for sec in doc.content.sections):
        return None
    return _warn("misc", f"Document {doc_index}: pas de section exercices/activities")

def _sequence_findings(category, base_findings, label_singular, label_sequence, numbers, duplicates, expected):
    findings = list(base_findings)
    details = []
    if duplicates:
        details.append(f"doublon {label_singular} {', '.join(str(n) for n in sorted(duplicates))}")
    if numbers != expected:
        details.append(label_sequence)
    if details:
        findings.append(_warn(category, f"{' et '.join(details)} : {numbers} (attendu: {expected})"))
    return findings

def _validate_exercise_sequence(slides: list[Slide]) -> list[Finding]:
    findings: list[Finding] = []
    numbers: list[int] = []
    seen: set[int] = set()
    duplicates: set[int] = set()

    for slide in slides:
        parsed = parse_exercise_title(slide.title)
        if parsed.status == "invalid":
            findings.append(_warn("sequence_exercises",
                f"slide {slide.id}: titre invalide '{slide.title}'", slide_id=slide.id))
            continue
        if parsed.status == "recoverable":
            findings.append(_warn("sequence_exercises",
                f"slide {slide.id}: titre non standard '{slide.title}' (numero={parsed.number})",
                slide_id=slide.id))
        if parsed.number in seen:
            duplicates.add(parsed.number)
        else:
            seen.add(parsed.number)
        numbers.append(parsed.number)

    if not numbers:
        return findings
    return _sequence_findings("sequence_exercises", findings, "exercice", "ordre physique incorrect",
                               numbers, duplicates, list(range(1, max(numbers) + 1)))

@dataclass
class ReferenceResolution:
    status:      Literal["valid", "recoverable", "invalid"]
    exercise_id: Optional[int]
    number:      Optional[int]
    findings:    list[Finding]

def resolve_exercise_reference(
    slide: Slide,
    section_id: int,
    parsed: TitleParseResult,
    ex_slides_by_id: dict[int, Slide],
    ex_num_to_id: dict[int, int],
) -> ReferenceResolution:
    kw = dict(section_id=section_id, slide_id=slide.id, field_name="exercise_id")
    exercise_id = slide.exercise_id
    id_ok = exercise_id is not None and exercise_id in ex_slides_by_id
    title_ok = parsed.status != "invalid" and parsed.number is not None
    title_resolvable = title_ok and parsed.number in ex_num_to_id
    ref_number = parse_exercise_title(ex_slides_by_id[exercise_id].title).number if id_ok else None

    if id_ok and title_ok:
        if ref_number == parsed.number:
            return ReferenceResolution("valid", exercise_id, parsed.number, [])
        mismatch = _warn("exercise_correction",
            f"slide {slide.id}: mismatch exercise_id->exercice {ref_number} / titre->exercice {parsed.number}", **kw)
        return ReferenceResolution("recoverable", exercise_id, ref_number, [mismatch])

    if not id_ok and title_resolvable:
        resolved_id = ex_num_to_id[parsed.number]
        if exercise_id is None:
            message = f"slide {slide.id}: exercise_id absent, titre -> exercice {parsed.number}"
        else:
            message = (f"slide {slide.id}: exercise_id={exercise_id} introuvable, "
                       f"titre -> exercice {parsed.number} (slide {resolved_id})")
        return ReferenceResolution("recoverable", resolved_id, parsed.number, [_warn("exercise_correction", message, **kw)])

    if id_ok and not title_ok:
        message = (f"slide {slide.id}: titre invalide '{slide.title}', "
                   f"exercise_id -> exercice {ref_number}")
        return ReferenceResolution("recoverable", exercise_id, ref_number, [_warn("exercise_correction", message, **kw)])

    message = (f"slide {slide.id}: reference non resolvable "
               f"(exercise_id={exercise_id!r}, titre='{slide.title}')")
    return ReferenceResolution("invalid", None, None, [_err("exercise_correction", message, **kw)])

def _validate_activities(sections: list[Section]) -> list[Finding]:
    ex_slides_by_id, ex_num_to_id = build_exercise_index(sections)

    findings: list[Finding] = []
    numbers: list[int] = []
    seen: set[int] = set()
    duplicates: set[int] = set()

    for sec in sections:
        if sec.kind != "activities":
            continue
        for slide in sec.slides:
            parsed = parse_exercise_title(slide.title)
            resolution = resolve_exercise_reference(slide, sec.id, parsed, ex_slides_by_id, ex_num_to_id)
            if resolution.status != "invalid":
                findings.extend(resolution.findings)
            if resolution.number is not None:
                if resolution.number in seen:
                    duplicates.add(resolution.number)
                else:
                    seen.add(resolution.number)
                numbers.append(resolution.number)

    if not numbers:
        return findings
    return _sequence_findings("exercise_correction", findings, "correction",
                               "sequence corrections incorrecte", numbers, duplicates,
                               list(range(1, max(numbers) + 1)))

def _has_valid_correction_block(slide: Slide) -> bool:
    return any(
        block.type == "correction"
        and getattr(getattr(block, "data", None), "label", None)
        and getattr(getattr(block, "data", None), "text", None)
        for block in (slide.blocks or [])
    )

def check_exercise_coverage(sections: list[Section]) -> CoverageResult:
    findings: list[Finding] = []
    rejected: list[tuple[Optional[int], Optional[int]]] = []

    ex_slides_by_id, ex_num_to_id = build_exercise_index(sections)
    covered: set[int] = set()

    for sec in sections:
        if sec.kind != "activities":
            continue
        for slide in sec.slides:
            kw = dict(section_id=sec.id, slide_id=slide.id, field_name="exercise_id")
            parsed = parse_exercise_title(slide.title)
            resolution = resolve_exercise_reference(slide, sec.id, parsed, ex_slides_by_id, ex_num_to_id)

            if resolution.status == "invalid":
                findings.append(_err("exercise_correction",
                    f"slide {slide.id}: reference exercice introuvable "
                    f"(exercise_id={slide.exercise_id!r}, titre='{slide.title}')", **kw))
                rejected.append((None, slide.id))
                continue

            exercise_id = resolution.exercise_id
            if not _has_valid_correction_block(slide):
                findings.append(_err("exercise_correction",
                    f"slide {slide.id}: bloc correction invalide "
                    f"(label/text manquant) exercice={exercise_id}", **kw))
                rejected.append((exercise_id, slide.id))
                continue

            covered.add(exercise_id)

    rejected_exercise_ids = {eid for eid, _ in rejected if eid is not None}
    for exercise_id, slide in ex_slides_by_id.items():
        if exercise_id not in covered and exercise_id not in rejected_exercise_ids:
            findings.append(_err("exercise_correction",
                f"exercice {exercise_id} '{slide.title}': sans correction",
                slide_id=exercise_id))
            rejected.append((exercise_id, None))

    return CoverageResult(findings=findings, rejected_pairs=rejected)


# ── Document parsing (schema validation with graceful degradation) ──────

@dataclass
class _ParsedDocument:
    document: Optional[Document]
    field_findings: list[Finding]
    pydantic_findings: list[Finding]


def _parse_document(raw: object, doc_index: int) -> _ParsedDocument:
    if not isinstance(raw, dict):
        finding = _err("document_fields", f"Document {doc_index}: entree invalide (objet JSON attendu)",
                       doc_index=doc_index)
        return _ParsedDocument(document=None, field_findings=[finding], pydantic_findings=[])

    field_findings = _validate_document_fields(raw, doc_index)
    already_reported = frozenset(f.field_name for f in field_findings if f.field_name)

    try:
        document = Document.model_validate(raw)
        return _ParsedDocument(document=document, field_findings=field_findings, pydantic_findings=[])
    except Exception as exc:
        pydantic_findings = _translate_pydantic_errors(exc, doc_index, skip_fields=already_reported)
        try:
            document = _build_partial(Document, raw)
        except Exception:
            document = None
        return _ParsedDocument(document=document, field_findings=field_findings, pydantic_findings=pydantic_findings)