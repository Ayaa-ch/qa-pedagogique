from __future__ import annotations

import json
import sys
from pathlib import Path

from correction import (
    _build_report_steps,
    correct,
    export_corrected_and_orphans,
)
from helpers import ensure_ollama_server

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
REPORT_DIR = RESULTS_DIR / "report"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _write_report(result, path: Path) -> None:
    lines = []
    for title, items, outcome in _build_report_steps(result):
        lines.append(f"\n=== {title} ===")
        if not items:
            lines.append("  (aucun)")
        for item in items:
            lines.append(f"- {item}")
        lines.append(f"Resultat: {outcome}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_feedback_report(result, path: Path, total_docs: int) -> None:
    from feedback import write_feedback_report

    if result.language_rejections:
        items = []
        for check in result.language_rejections:
            items.append(
                f"Document {(getattr(check, 'doc_index', None) or 0) + 1} : "
                f"langue attendue '{getattr(check, 'expected', '?')}', rejete"
            )
            items.extend(str(f) for f in getattr(check, "findings", []) or [])
        feedback = write_feedback_report(
            steps=[(
                "REJET LANGUE (>= 30% hors langue attendue)",
                items or ["(detail non fourni)"],
                "lot entier rejete : aucune correction effectuee",
            )],
            counts={"documents rejetes (langue)": len(result.language_rejections)},
            meta={
                "documents_total": total_docs,
                "documents_corriges": 0,
                "documents_titles": [doc.title for doc in result.documents],
                "rejets": f"{len(result.language_rejections)} document(s) rejete(s) (langue)",
            },
            directory=REPORT_DIR,
        )
    else:
        feedback = write_feedback_report(
            steps=_build_report_steps(result),
            counts={
                "paires rejetees (orphelins + doublons + hors-sujet)": len(result.rejected_pairs),
                "anomalies de structure signalees": len(result.structure_findings),
                "incoherences de sequence corrigees": len(result.resolved_findings),
                "reponses generees (NEEDS_GENERATION)": len(result.content_generated),
                "labels realignes (NEEDS_RESTRUCTURE)": len(result.restructure_fixed),
                "cas en verification manuelle (validator)": len(result.validator_manual_review),
                "titres a verifier manuellement (generation)": len(result.generation_findings),
                "erreurs de rendu corrigees": len(result.render_resolved_findings),
                "erreurs de rendu restantes": len(result.render_verification_findings),
                "points restants (verification)": len(result.verification_findings),
                "avertissements de langue": len(result.language_warnings),
            },
            meta={
                "documents_total": total_docs,
                "documents_corriges": len(result.documents),
                "documents_titles": [doc.title for doc in result.documents],
                "rejets": (
                    f"{len(result.parsing_rejected)} au parsing, "
                    f"{len(result.language_rejections)} langue"
                ),
            },
            directory=REPORT_DIR,
        )
    if feedback["pdf"]:
        print(f"PDF du rapport d'analyse: {feedback['pdf']}")
    if feedback["note"]:
        print(f"[WARN] {feedback['note']}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python launch.py {nom.json}")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Fichier introuvable: {path}")
        return 1

    try:
        ensure_ollama_server()
    except Exception as exc:
        print(f"Ollama indisponible: {exc}")
        return 1

    with open(path, encoding="utf-8") as f:
        original_data = json.load(f)

    raw_documents = original_data if isinstance(original_data, list) else [original_data]
    total_docs = len(raw_documents)
    result = correct(raw_documents)

    _ensure_dirs()
    out_path = DATA_DIR / (path.stem + "_corrected" + path.suffix)
    report_path = REPORT_DIR / "rapport_corrections.txt"

    if result.language_rejections:
        _write_report(result, report_path)
        print(f"Rejet langue: aucun fichier corrige genere.")
        print(f"Rapport detaille ecrit dans: {report_path}")
        try:
            _write_feedback_report(result, path, total_docs)
        except Exception as exc:
            print(f"Rapport LLM impossible: {exc}")
            return 1
        return 0

    corrected_raw, orphans_raw = export_corrected_and_orphans(raw_documents, result)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected_raw, f, ensure_ascii=False, indent=2)

    if orphans_raw:
        orphans_path = DATA_DIR / (path.stem + "_orphans" + path.suffix)
        with open(orphans_path, "w", encoding="utf-8") as f:
            json.dump(orphans_raw, f, ensure_ascii=False, indent=2)
        print(f"Orphelins ecrits dans: {orphans_path}")

    _write_report(result, report_path)
    print(f"Fichier corrige ecrit dans: {out_path}")
    print(f"Rapport detaille ecrit dans: {report_path}")
    try:
        _write_feedback_report(result, path, total_docs)
    except Exception as exc:
        print(f"Rapport LLM impossible: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
