"""
Vérification de cohérence :
1. doublon   :Exercice i ↔ Exercice j → reject.
2. Titre global ↔ chaque slide du résumé
   - Détection des slides de résumé hors sujet → reject("resume", slide_id).
   - Si aucun résumé exploitable → review("resume").
3. Titre global ↔ chaque exercice restant → reject("exercise") et son correction .
4. Titre de chaque slide de résumé ↔ contenu de cette slide() → needs_generation("slide_title", slide_id).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from parser import Document, Slide
from helpers import exercise_text, resume_slide_texts, slide_resume_text, strip_markup

# Model
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_TOPICAL_MATCH_THRESHOLD = 0.35  #lien thematique large
_DUPLICATE_SIMILARITY_THRESHOLD = 0.92  # quasi-identique pour les exercices 

_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

#embeddings
def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))
    return _get_model().encode(texts, normalize_embeddings=True)

#similarity
def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0))
    return embeddings @ embeddings.T

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0

#fonction de thematique 
def topical_match(text_a: str, text_b: str, threshold: float = _TOPICAL_MATCH_THRESHOLD) -> bool:
    """Coherence thematique large : 'ca parle du meme sujet ?'"""
    text_a, text_b = strip_markup(text_a), strip_markup(text_b)
    if not text_a or not text_b:
        return False
    emb = embed_texts([text_a, text_b])
    return cosine_similarity(emb[0].tolist(), emb[1].tolist()) >= threshold

# ── Types de findings ────────────────────────────────────────────────────
CheckKind = Literal["resume", "off_exercise_pair", "slide_title", "dup_exercise_pair"]
Verdict = Literal["needs_generation", "reject", "review"]

@dataclass
class CoherenceFinding:
    kind: CheckKind
    verdict: Verdict
    doc_title: str
    slide_id: Optional[int] = None
    other_slide_id: Optional[int] = None
    detail: Optional[str] = None

    def __str__(self) -> str:
        loc = f" slide={self.slide_id}"
        loc += f",{self.other_slide_id}" if self.other_slide_id is not None else ""
        return f"[{self.kind}] {self.verdict}{loc} — {self.detail or ''}".strip()

# ── Etape 1 : Exercice i <-> Exercice j (doublons) ───────────────────────

def find_exercise_duplicates(doc: Document, corr_by_ex_id: dict[int, Slide]) -> list[CoherenceFinding]:
    """Doublons exercice<->exercice : renvoie la paire (exercice duplique, sa correction) a rejeter."""
    all_ex_slides = [
        s for sec in doc.content.sections if sec.kind == "exercises" for s in sec.slides
    ]
    texts = [strip_markup(exercise_text(s)) for s in all_ex_slides]
    valid_idx = [i for i, t in enumerate(texts) if t]
    findings: list[CoherenceFinding] = []

    if len(valid_idx) < 2:
        return findings

    embeddings = embed_texts([texts[i] for i in valid_idx])
    sims = cosine_matrix(embeddings)

    removed: set[int] = set()  # indices (dans valid_idx) deja rejetes
    n = len(valid_idx)
    for a in range(n):
        if a in removed:
            continue
        for b in range(a + 1, n):
            if b in removed:
                continue
            if sims[a, b] >= _DUPLICATE_SIMILARITY_THRESHOLD:
                ex_a = all_ex_slides[valid_idx[a]]
                ex_b = all_ex_slides[valid_idx[b]]
                corr_b = corr_by_ex_id.get(ex_b.id)
                findings.append(CoherenceFinding(
                    "dup_exercise_pair", "reject", doc.title,
                    slide_id=ex_b.id,
                    other_slide_id=corr_b.id if corr_b is not None else None,
                    detail=f"'{ex_a.title}' et '{ex_b.title}' ont un contenu identique — '{ex_b.title}' rejete avec sa correction",
                ))
                removed.add(b)

    return findings


def diagnose_document(doc: Document, corr_by_ex_id: dict[int, Slide]) -> list[CoherenceFinding]:
    findings = find_exercise_duplicates(doc, corr_by_ex_id)
    title_findings = check_title_coherence(doc, corr_by_ex_id)
    findings.extend(title_findings)
    rejected_res = {f.slide_id for f in title_findings
                    if f.kind == "resume" and f.slide_id is not None}
    all_res_ids = {
        s.id
        for sec in doc.content.sections if sec.kind == "resume"
        for s in sec.slides
    }
    coherent_res = all_res_ids - rejected_res
    findings.extend(check_slide_title_coherence(doc, coherent_res))
    return findings


# ── Etape 2 : Titre <-> Resume <-> chaque Exercice restant ───────

def check_title_coherence(doc: Document, corr_by_ex_id: dict[int, Slide]) -> list[CoherenceFinding]:
    sections = doc.content.sections
    resume_texts = resume_slide_texts(sections)
    all_ex_slides = [s for sec in sections if sec.kind == "exercises" for s in sec.slides]
    exercise_texts = {s.id: exercise_text(s) for s in all_ex_slides}

    findings: list[CoherenceFinding] = []

    if not resume_texts:
        findings.append(CoherenceFinding(
            "resume", "review", doc.title,
            detail="resume absent ou vide — redaction manuelle requise",
        ))
        return findings

    clean_title = strip_markup(doc.title)
    res_items = [(sid, strip_markup(t)) for sid, t in resume_texts.items() if t]
    ex_items = [(eid, strip_markup(t)) for eid, t in exercise_texts.items() if t]

    all_res_ids = list(resume_texts.keys())
    all_ex_ids = list(exercise_texts.keys())

    if not clean_title or len(res_items) + len(ex_items) == 0:
        # Pas de titre exploitable, ou ni resume ni exercice exploitable (tout est incoherent)
        incoherent_res_ids, incoherent_ex_ids = all_res_ids, all_ex_ids
    else:
        labels: list[str] = ["title"]
        texts: list[str] = [clean_title]
        for sid, t in res_items:
            labels.append(f"res:{sid}")
            texts.append(t)
        for eid, t in ex_items:
            labels.append(f"ex:{eid}")
            texts.append(t)

        embeddings = embed_texts(texts)
        sims = cosine_matrix(embeddings)
        title_row = sims[0]

        incoherent_res_ids: list[int] = []
        incoherent_ex_ids: list[int] = []
        res_ids_checked: set[int] = set()
        exercise_ids_checked: set[int] = set()

        for i, label in enumerate(labels):
            if i == 0:
                continue
            if label.startswith("res:"):
                sid = int(label.split(":", 1)[1])
                res_ids_checked.add(sid)
                if title_row[i] < _TOPICAL_MATCH_THRESHOLD:
                    incoherent_res_ids.append(sid)
            elif label.startswith("ex:"):
                eid = int(label.split(":", 1)[1])
                exercise_ids_checked.add(eid)
                if title_row[i] < _TOPICAL_MATCH_THRESHOLD:
                    incoherent_ex_ids.append(eid)

        # Slides/exercices sans texte exploitable : incoherents par defaut.
        incoherent_res_ids.extend(sid for sid in all_res_ids if sid not in res_ids_checked)
        incoherent_ex_ids.extend(eid for eid in all_ex_ids if eid not in exercise_ids_checked)

    resume_slide_lookup = {
        slide.id: slide
        for sec in sections if sec.kind == "resume"
        for slide in sec.slides
    }
    for sid in incoherent_res_ids:
        slide = resume_slide_lookup.get(sid)
        title = slide.title if slide else None
        findings.append(CoherenceFinding(
            "resume", "reject", doc.title, slide_id=sid,
            detail=f"slide de resume '{title}' hors-sujet par rapport au titre '{doc.title}'" if title else None,
        ))

    slides_by_id = {s.id: s for s in all_ex_slides}
    for eid in incoherent_ex_ids:
        slide = slides_by_id.get(eid)
        title = slide.title if slide else None
        corr = corr_by_ex_id.get(eid)
        findings.append(CoherenceFinding(
            "off_exercise_pair", "reject", doc.title,
            slide_id=eid,
            other_slide_id=corr.id if corr is not None else None,
            detail=f"'{title}' hors-sujet par rapport au titre '{doc.title}'" if title else None,
        ))

    return findings


# ── Etape 3 : titre de chaque slide de resume <-> son propre contenu ─────

def check_slide_title_coherence(doc: Document, coherent_res_ids: set[int]) -> list[CoherenceFinding]:
    """Titre de chaque slide de resume <-> son propre contenu"""
    findings: list[CoherenceFinding] = []
    if not coherent_res_ids:
        return findings

    entries: list[tuple[int, str, str]] = []  # (id, titre, contenu)
    for sec in doc.content.sections:
        if sec.kind != "resume":
            continue
        for slide in sec.slides:
            if slide.id not in coherent_res_ids:
                continue
            title = strip_markup(slide.title or "")
            text = strip_markup(slide_resume_text(slide))
            if title and text:
                entries.append((slide.id, title, text))

    if not entries:
        return findings

    flat_texts: list[str] = []
    for _, title, text in entries:
        flat_texts.append(title)
        flat_texts.append(text)

    embeddings = embed_texts(flat_texts)

    slide_lookup = {
        slide.id: slide
        for sec in doc.content.sections if sec.kind == "resume"
        for slide in sec.slides
    }
    for i, (slide_id, _, _) in enumerate(entries):
        title_emb = embeddings[2 * i]
        text_emb = embeddings[2 * i + 1]
        sim = float(np.dot(title_emb, text_emb))
        if sim < _TOPICAL_MATCH_THRESHOLD:
            slide = slide_lookup.get(slide_id)
            title = slide.title if slide else None
            findings.append(CoherenceFinding(
                "slide_title", "needs_generation", doc.title, slide_id=slide_id,
                detail=f"le titre du slide '{title}' ne correspond pas a son propre contenu" if title else None,
            ))

    return findings
