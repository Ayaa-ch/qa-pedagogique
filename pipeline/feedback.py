from __future__ import annotations

import re
import time
from pathlib import Path

import requests
import markdown as md_lib
from xhtml2pdf import pisa

from helpers import ensure_ollama_server

RAPPORT_PDF = "rapport_analyse.pdf"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_NUM_CTX = 8192   

def call_ollama_text(messages: list[dict], model: str = OLLAMA_MODEL) -> tuple[str, float]:
    """Appelle l'API Ollama et retourne la réponse et le temps d'exécution."""
    started_at = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.4, "num_ctx": OLLAMA_NUM_CTX},
        },
    )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return response.json()["message"]["content"], elapsed_ms

_SYSTEM_PROMPT = (
    "Tu es un expert en relecture et correction pédagogique. "
    "Tu rédiges en français un rapport d'analyse naturel et lisible, "
    "dans un ton professionnel et direct, sans formules toutes faites "
    "ni recopiage mécanique. Tu expliques ce que le pipeline a fait, "
    "pourquoi il l'a fait et ce que cela implique pour l'auteur. "
    "Tu restes fidèle aux faits fournis (exercice/correction, "
    "marqueurs, blocs, chiffres), mais tu les présentes sous forme "
    "d'explications, pas de tags techniques."
)


_ECRITURE_PROMPT = """Rédige un rapport Markdown structuré qui **explique** ce qui a été fait, sans forme imposée. Ne reprends pas mécaniquement les étapes fournies : synthétise-les dans tes propres mots.

1. **Ce qui a été corrigé** – Raconte ce que le pipeline a modifié dans le document, en prose : réponses générées pour des marqueurs manquants (indique l'exercice/correction concerné, le **marqueur exact** généré, ex. « 2.a) », « 4.b) », et la question traitée), restructurations (une réponse globale découpée en sous-réponses alignées sur les marqueurs de l'énoncé, en citant ces marqueurs), réordonnancements, titres régénérés, rejets (orphelins, doublons, hors-sujet). Explique pourquoi chaque action a été nécessaire, en restant simple et concret. Pour le rendu, indique combien d'erreurs LaTeX/HTML/Markdown ont été corrigées et ce qu'elles avaient en commun (ex. : sauts de ligne dans les formules $$, blocs non fermés). Le détail bloc par bloc est dans `rapport_corrections.txt` ; tu n'as pas à le recopier ici.

2. **Origine des problèmes** – Explique les causes probables des erreurs rencontrées (mise en forme des formules, `\\n` littéraux, structure des corrections, marqueurs manquants), regroupées par cause pour éviter les répétitions, avec les chiffres correspondants.

3. **Conseils aux auteurs** – Recommandations pratiques, priorisées par impact, pour éviter ces problèmes à l'avenir. En puces, sans répéter les explications précédentes.

4. **Ce qui reste à faire** – Vérifications manuelles restantes (cas soumis à revue, marqueurs encore manquants), points de vigilance, et renvoi à `rapport_corrections.txt` pour le détail. **Liste explicitement chaque marqueur dont la réponse a été générée par le LLM** (ex. « 4.a) », « 2.a) ») afin qu'un relecteur humain puisse vérifier ces réponses précises. Résume le nombre global de cas restants.

**Important** :
- Ne juge pas le contenu scientifique ou pédagogique.
- Le titre principal doit être le titre du document (fourni dans l'en-tête). Sinon, utilise un titre générique.
- Sois synthétique : chaque section se concentre sur l'essentiel ; ne répète pas la même information d'une section à l'autre.
- Ne commence pas par une phrase d'introduction générale.
- Ne termine pas par une conclusion.

Fournis uniquement le texte du rapport, sans en-tête ni pied de page supplémentaires.
"""

_NOTE_LIMITES = """## NOTE — Ce que ce rapport ne vérifie pas

Ce pipeline ne vérifie pas : la justesse scientifique ou pédagogique des
questions (une question incorrecte n'est pas détectée si elle existe), les
erreurs grammaticales, orthographiques ou typographiques du contenu, ni la
validité des réponses générées. Une relecture humaine reste indispensable.
"""

def build_feedback_prompt(
    steps: list[tuple[str, list[str], str]],
    counts: dict[str, int],
    meta: dict,
) -> str:
    """Construit le prompt complet à partir des données d'entrée."""
    lines = []
    titles = meta.get("documents_titles") or []
    if titles:
        lines.append("# " + " — ".join(titles))
    else:
        lines.append("# RAPPORT DU PIPELINE DE CORRECTION")
    lines.append("")
    lines.append(f"- Date : {meta.get('date', '')}")
    lines.append(f"- Documents traités : {meta.get('documents_total', 0)}")
    lines.append(f"- Documents corrigés : {meta.get('documents_corriges', 0)}")
    if meta.get("rejets"):
        lines.append(f"- Rejets : {meta['rejets']}")
    lines.append("")
    lines.append("## Étapes et résultats")
    if steps:
        for title, items, outcome in steps:
            lines.append(f"\n### {title}")
            lines.append(f"Résultat : {outcome}")
            if items:
                lines.append("Détails :")
                for item in items:
                    lines.append(f"- {item}")
    else:
        lines.append("(aucune étape fournie)")
    lines.append("")
    lines.append("## Compteurs globaux")
    if counts:
        for key, value in counts.items():
            lines.append(f"- {key} : {value}")
    else:
        lines.append("(aucun compteur)")
    lines.append("")
    lines.append(_ECRITURE_PROMPT)
    return "\n".join(lines)

def generate_feedback(
    steps: list[tuple[str, list[str], str]],
    counts: dict[str, int],
    meta: dict,
    model: str = OLLAMA_MODEL,
) -> str:
    """Génère le rapport via Ollama."""
    prompt = build_feedback_prompt(steps, counts, meta)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    content, _elapsed = call_ollama_text(messages, model=model)
    return content.strip()

def render_pdf(markdown_text: str, output_path: Path) -> None:
    """Convertit le Markdown en PDF avec un style professionnel."""
    html = md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    styled = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body { font-family: Helvetica; font-size: 11px; color: #222; line-height: 1.45; }"
        "h1 { font-size: 18px; color: #1a3d6d; border-bottom: 2px solid #1a3d6d; padding-bottom: 4px; }"
        "h2 { font-size: 14px; color: #1a3d6d; margin-top: 14px; }"
        "h3 { font-size: 12px; color: #333; margin-top: 10px; }"
        "table { border-collapse: collapse; width: 100%; margin: 6px 0; }"
        "th, td { border: 1px solid #bbb; padding: 3px 5px; font-size: 10px; }"
        "th { background-color: #eef2f7; }"
        "blockquote { color: #666; border-left: 3px solid #ccc; margin-left: 0; padding-left: 8px; }"
        "code { font-family: Courier; font-size: 9px; background: #f4f4f4; }"
        "</style></head><body>" + html + "</body></html>"
    )

    with open(output_path, "wb") as f:
        status = pisa.CreatePDF(styled, dest=f, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"pisa.CreatePDF a retourné err={status.err}")

def write_feedback_report(
    steps: list[tuple[str, list[str], str]],
    counts: dict[str, int],
    meta: dict,
    directory: str | Path = ".",
) -> dict:
    """
    Fonction principale : génère le rapport Markdown via Ollama, le convertit en PDF
    et retourne le chemin du PDF et une éventuelle note.
    """
    directory = Path(directory)
    ensure_ollama_server()  

    content = generate_feedback(steps, counts, meta)

    content = content.strip() + "\n\n" + _NOTE_LIMITES
    content += "\n\n> Détail de chaque bloc modifié : consulter `rapport_corrections.txt`."

    pdf_path = directory / RAPPORT_PDF

    pdf_ok = False
    note = ""
    try:
        render_pdf(content, pdf_path)
        pdf_ok = True
    except Exception as exc:
        note = (note + "; " if note else "") + f"PDF non généré : {exc}"

    return {
        "pdf": str(pdf_path) if pdf_ok else None,
        "note": note,
    }
