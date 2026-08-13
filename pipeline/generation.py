from __future__ import annotations

import json
from typing import Optional

import requests

from helpers import parse_llm_json

# Appel modele 

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL_NAME = "qwen2.5:7b"


_NUM_CTX = 16384  # aligne sur validator.OLLAMA_NUM_CTX.

def _call_llm(system_prompt: str, user_prompt: str, num_ctx: int = _NUM_CTX) -> str:
    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": _MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json", 
            "options": {"temperature": 0.0, "num_ctx": num_ctx},
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def generate_slide_title(doc_title: str, slide_blocks: list[str], expected_lang: str = "fr") -> dict:
    system_prompt = (
    "Tu génères le titre d'une diapositive académique.\n\n"
    "Avant de répondre, identifie mentalement :\n"
    "1. Le sujet principal.\n"
    "2. Ce que le contenu explique\n\n"
    "Le titre doit nommer cette idée centrale sous forme d'un groupe nominal (3 à 6 mots max), "
    "sans recopier des mots-clés mettre l'accent sur tous les détails \n"
    "Interdit : infinitif, impératif.\n\n"
    f"Rédige le titre en {expected_lang}.\n\n"
    'Réponse : {"title": "<titre>"}'
    )
    payload = {"titre_document": doc_title, "blocs_du_slide": slide_blocks}
    parsed = parse_llm_json(_call_llm(system_prompt, json.dumps(payload, ensure_ascii=False)))
    if parsed is None or not isinstance(parsed, dict) or "title" not in parsed:
        raise ValueError("reponse LLM illisible (JSON invalide)")
    return {"title": parsed["title"]}

# NEEDS_RESTRUCTURE

def restructure_corrections(
    markers_a_traiter: list[str],
    questions_by_marker: dict[str, str],
    global_text: str,
) -> Optional[dict[str, str]]:
    """Eclate une correction globale en une reponse par sous-marqueur.

    Entrees calquees sur la sortie de validator_v2 :
    - markers_a_traiter = entry["markers_a_traiter"]
    - global_text = texte du bloc correction label = entry["correction_label"]
    - questions_by_marker = dict marqueur -> question
    """
    if not markers_a_traiter:
        return None

    sub_questions = [
        {"marqueur": m, "question": questions_by_marker.get(m, "")}
        for m in markers_a_traiter
    ]

    system_prompt = (
        "Tu es un assistant pedagogique. UNE seule reponse de correction "
        "globale couvre plusieurs sous-questions a la fois. Tu dois decouper "
        "cette reponse pour attribuer a CHAQUE sous-marqueur UNIQUEMENT la "
        "partie qui repond a sa question.\n\n"
        "OBJECTIF : associer a chaque marqueur fourni le texte de reponse "
        "qui lui correspond, extrait de la reponse globale.\n"
        "- Reprends le texte tel quel, sans reformulation (ni ajout, ni "
        "suppression, ni modification).\n"
        "- Un marqueur sans contenu correspondant dans la reponse globale "
        "doit avoir une chaine vide \"\".\n\n"
        'Reponds uniquement avec un objet JSON de la forme '
        '{"attribution": {"<marqueur>": "<texte>", ...}}, avec exactement '
        "une entree par marqueur recu."
    )
    user_prompt = json.dumps(
        {
            "sous_questions": sub_questions,
            "reponse_globale": global_text,
        },
        ensure_ascii=False,
    )
    try:
        parsed = parse_llm_json(_call_llm(system_prompt, user_prompt))
    except Exception:
        return None
    if parsed is None or not isinstance(parsed, dict):
        return None

    attribution = parsed.get("attribution", {})
    if not isinstance(attribution, dict):
        return None
    result: dict[str, str] = {}
    for m in markers_a_traiter:
        text = attribution.get(m)
        if not isinstance(text, str):
            return None
        result[m] = text.strip()
    return result


def generate_missing_correction_blocks(
    missing_entries: list[dict],
    questions_by_marker: dict[str, str],
    existing_blocks: list[dict],
    exercise_text: str,
    expected_lang: str = "fr",
) -> list[dict]:
    marqueurs_a_completer = [
        {
            "marqueur": e["marqueur"],
            "question": questions_by_marker.get(e["marqueur"], ""),
            "raison": e.get("reason", ""),
        }
        for e in missing_entries
        if e.get("marqueur")
    ]
    if not marqueurs_a_completer:
        return []

    system_prompt = (
        "Tu es un assistant pedagogique. Tu recois l'enonce complet d'un "
        "exercice, les reponses de correction deja presentes, et la liste "
        "EXACTE des marqueurs de questions SANS reponse pour lesquels tu "
        "dois rediger une reponse.\n\n"
        "Redige UNE reponse par marqueur fourni dans `marqueurs_a_completer`, "
        "en repondant precisement a la question correspondante.\n\n"
        "GABARIT STRUCTUREL OBLIGATOIRE (identique aux corrections DCC existantes) :\n"
        "Contraintes strictes :\n"
        "- Un element JSON par marqueur, dans le MEME ordre que "
        "`marqueurs_a_completer`, avec le label recopie tel quel.\n"
        "- Ne reecris PAS les reponses deja presentes dans `reponses_existantes`.\n"
        "- Reste factuel et rigoureux, meme niveau de detail que le reste "
        "de la correction.\n"
        "- Utilise UNIQUEMENT <br> pour les sauts de ligne (JAMAIS \\n).\n"
        "- Conserve les conventions : gras **...**, LaTeX inline $...$ "
        "et affiche $$...$$.\n"
        "- Commence chaque reponse de meme forme que les corrections "
        "deja presentes.\n"
        "- Avant de rediger, observe comment les reponses DEJA PRESENTES "
        "sont structurees et reproduis le meme gabarit structurel pour "
        "chaque reponse manquante.\n"
        f"- Redige chaque reponse en {expected_lang}.\n\n"
        'Reponds uniquement avec un objet JSON de la forme '
        '{"reponses": [{"label": "<marqueur>", "text": "<texte>"}, ...]}, '
        "avec exactement un element par marqueur recu, sans aucun autre texte."
    )
    user_prompt = json.dumps(
        {
            "enonce": exercise_text,
            "reponses_existantes": existing_blocks,
            "manques_identifies": [e["raison"] for e in marqueurs_a_completer],
            "marqueurs_a_completer": marqueurs_a_completer,
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_json(_call_llm(system_prompt, user_prompt))
    if parsed is None or not isinstance(parsed, dict):
        return []
    by_label = {r["label"]: r["text"] for r in parsed.get("reponses", [])}
    return [
        {"label": e["marqueur"], "text": by_label.get(e["marqueur"], "")}
        for e in missing_entries
        if e.get("marqueur")
    ]
