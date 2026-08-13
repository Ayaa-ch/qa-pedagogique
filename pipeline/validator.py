from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal, Optional

import requests

from label_sequence import build_marker_pairs, classify_marker_pairs
from parser import Slide
from helpers import clean_text, enonce_text, parse_llm_json, prompt_text

ContentStatus = Literal["COMPLETE", "INCOMPLETE"]
MarkerStatus = Literal["OK", "NEEDS_GENERATION", "NEEDS_RESTRUCTURE"]

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_NUM_CTX = 16384

# 1. Appel LLM (format JSON)
def call_qwen(
    messages: list[dict] | str,
    model: str = OLLAMA_MODEL,
    num_ctx: int = OLLAMA_NUM_CTX,
) -> tuple[str, float]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    started_at = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": num_ctx},
        },
    )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return response.json()["message"]["content"], elapsed_ms

# 2. Extraction exercice / correction
def _correction_blocks(slide: Slide) -> list[tuple[str, str]]:
    blocks = []
    for b in slide.blocks:
        if b.type == "correction" and b.data.text:
            label = (b.data.label or "").strip()
            text = clean_text(b.data.text)
            if text:
                blocks.append((label, text))
    return blocks

def _prepare_inputs(
    exercise_slide: Slide, correction_slide: Slide
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    prompt = prompt_text(exercise_slide)
    correction_blocks = _correction_blocks(correction_slide)
    correction_labels = [lbl for lbl, _ in correction_blocks]
    correction_texts = [txt for _, txt in correction_blocks]

    if not prompt:
        return [], {}, correction_labels, correction_texts

    pairs = build_marker_pairs(prompt, correction_labels, correction_texts)
    exercise_markers = [p["marqueur"] for p in pairs["marqueurs_questions"]]
    questions_by_marker = {p["marqueur"]: p["question"] for p in pairs["marqueurs_questions"]}
    return exercise_markers, questions_by_marker, correction_labels, correction_texts

# 3. Vérification LLM de couverture d'une correction globale
def llm_check_global_coverage(
    sub_questions: list[tuple[str, str]], global_answer: str, llm_call=call_qwen, enonce: str = ""
) -> tuple[bool, list[str], str]:
    prompt_system = """\
Tu es un assistant qui vérifie si UNE réponse de correction globale répond à
PLUSIEURS sous-questions à la fois.

Tu reçois l'énoncé de l'exercice (contexte), une liste de sous-questions (avec
leur marqueur) et le texte de la réponse globale de correction. Détermine si
CHAQUE sous-question trouve une réponse explicite et correcte quelque part dans
ce texte global.

SORTIE UNIQUEMENT CE JSON :
{"all_covered": true|false, "missing_markers": ["..."], "reason": "..."}
"""
    q_lines = "\n".join(f"- [{m}] {q}" for m, q in sub_questions)
    enonce_part = f"ENONCÉ DE L'EXERCICE :\n{enonce}\n\n" if enonce else ""
    prompt_user = f"""{enonce_part}SOUS-QUESTIONS :
{q_lines}

RÉPONSE GLOBALE DE CORRECTION :
{global_answer}
"""
    reply, _ = llm_call([
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user},
    ])
    data = parse_llm_json(reply)
    if data is None:
        return False, [m for m, _ in sub_questions], "JSON invalide"
    all_covered = bool(data.get("all_covered", False))
    missing = data.get("missing_markers") or []
    return all_covered, missing, data.get("reason", "")

# 4. Résultats
@dataclass
class MarkerResult:
    marker: str
    status: MarkerStatus
    reason: str
    related_markers: list[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    exercise_id: int
    correction_id: int
    content_status: ContentStatus
    marker_results: list[MarkerResult] = field(default_factory=list)
    need_generation: list[dict] = field(default_factory=list)
    need_restructure: list[dict] = field(default_factory=list)
    content_detail: Optional[str] = None
    response_time_ms: float = 0.0

# 5. Validation : labels déterministes (MISSING -> NEEDS_GENERATION),
#    + vérification LLM de couverture pour les corrections globales (-> NEEDS_RESTRUCTURE)
def validate(
    exercise_slide: Slide,
    correction_slide: Slide,
    llm_call=call_qwen,
) -> ValidationResult:
    started = time.perf_counter()

    exercise_markers, questions_by_marker, correction_labels, correction_texts = _prepare_inputs(
        exercise_slide, correction_slide
    )

    if not questions_by_marker:
        return ValidationResult(
            exercise_id=exercise_slide.id,
            correction_id=correction_slide.id,
            content_status="INCOMPLETE",
            content_detail="aucune question extraite",
        )
    if not correction_labels:
        return ValidationResult(
            exercise_id=exercise_slide.id,
            correction_id=correction_slide.id,
            content_status="INCOMPLETE",
            content_detail="aucune correction extraite",
        )

    correction_text_by_label = dict(zip(correction_labels, correction_texts))
    enonce = enonce_text(exercise_slide)
    groups = classify_marker_pairs(exercise_markers, correction_labels)

    marker_results: list[MarkerResult] = []
    need_generation: list[dict] = []
    need_restructure: list[dict] = []

    for group in groups:
        if group.kind in ("MISSING", "PARTIAL_MISSING"):
            for m in group.missing_markers:
                marker_results.append(MarkerResult(
                    m, "NEEDS_GENERATION", "marqueur manquant après normalisation",
                ))
                need_generation.append({
                    "marqueur": m,
                    "reason": "marqueur manquant après normalisation",
                    "exercise_id": exercise_slide.id,
                })
            for ex_marker, corr_label in group.matched_pairs:
                marker_results.append(MarkerResult(
                    ex_marker, "OK", f"corrigé par [{corr_label}]",
                ))
            continue

        if group.kind == "MATCHED":
            for ex_marker, corr_label in group.matched_pairs:
                marker_results.append(MarkerResult(
                    ex_marker, "OK", f"corrigé par [{corr_label}]",
                ))
            continue

        if group.kind == "GLOBAL_VS_SOUSQ":
            sub_questions = [(m, questions_by_marker.get(m, "")) for m in group.exercise_markers]
            global_answer = correction_text_by_label.get(group.correction_label, "")
            all_covered, missing_sub, reason = llm_check_global_coverage(
                sub_questions, global_answer, llm_call, enonce
            )
            if all_covered:
                marker_results.append(MarkerResult(
                    marker=group.correction_label,
                    status="NEEDS_RESTRUCTURE",
                    reason=reason or "réponse globale complète, à éclater par sous-marqueur",
                    related_markers=group.exercise_markers,
                ))
                need_restructure.append({
                    "correction_label": group.correction_label,
                    "markers_a_traiter": group.exercise_markers,
                    "exercise_id": exercise_slide.id,
                })
            else:
                missing = missing_sub or group.exercise_markers
                for m in missing:
                    marker_results.append(MarkerResult(
                        m, "NEEDS_GENERATION",
                        reason or "sous-question non couverte par la réponse globale",
                    ))
                    need_generation.append({
                        "marqueur": m,
                        "reason": reason or "sous-question non couverte par la réponse globale",
                        "exercise_id": exercise_slide.id,
                    })
            continue

    content_status: ContentStatus = (
        "INCOMPLETE" if (need_generation or need_restructure) else "COMPLETE"
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    return ValidationResult(
        exercise_id=exercise_slide.id,
        correction_id=correction_slide.id,
        content_status=content_status,
        marker_results=marker_results,
        need_generation=need_generation,
        need_restructure=need_restructure,
        response_time_ms=elapsed_ms,
    )
