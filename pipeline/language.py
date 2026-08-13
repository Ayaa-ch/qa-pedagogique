import yaml
import fasttext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from parser import Document
from helpers import block_texts, strip_markup


@dataclass
class LanguageFinding:
    severity: Literal["error", "warning"]
    category: str
    message: str
    block_id: Optional[int] = None
    slide_id: Optional[int] = None

    def __str__(self) -> str:
        prefix = "[ERROR]" if self.severity == "error" else "[WARNING]"
        return f"{prefix} {self.message}"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RULES = yaml.safe_load((_PROJECT_ROOT / "rules" / "language.yaml").read_text(encoding="utf-8"))
_TOPIC_MAP = {s["topic"]: s["expected_lang"] for s in _RULES["subjects"]}

_MODEL = fasttext.load_model(str(_PROJECT_ROOT / "models" / "lid.176.bin"))


LANG_ERROR_THRESHOLD = 0.30
MIN_CHARS            = 20


@dataclass
class LanguageCheckResult:
    expected: str
    rejected: bool
    findings: list[LanguageFinding]
    doc_index: Optional[int] = None


def _expected_lang(topic: Optional[str]) -> str:
    if topic is None:
        return "fr"
    return _TOPIC_MAP.get(topic, "fr")


def _detect(text: str) -> str:
    # fasttext ne supporte pas les newlines
    labels, _ = _MODEL.predict(text.replace("\n", " "), k=1)
    return labels[0].replace("__label__", "")


def check_document_language(doc: Document, doc_index: Optional[int] = None) -> LanguageCheckResult:
    topic = doc.topic or None
    expected = _expected_lang(topic)
    block_warnings: list[LanguageFinding] = []
    total = wrong = 0

    for sec in doc.content.sections:
        for slide in sec.slides:
            for block in slide.blocks:
                for raw in block_texts(block):
                    clean = strip_markup(raw)
                    if len(clean) < MIN_CHARS:
                        continue
                    total += 1
                    detected = _detect(clean)
                    if detected != expected:
                        wrong += 1
                        block_warnings.append(LanguageFinding(
                            severity="warning",
                            category="language",
                            message=f"langue inattendue dans le bloc {block.id} de la slide {slide.id}: "
                                    f"detecte '{detected}' au lieu de '{expected}'",
                            block_id=block.id,
                            slide_id=slide.id,
                        ))

    ratio = (wrong / total) if total else 0.0
    rejected = total > 0 and ratio >= LANG_ERROR_THRESHOLD
    if rejected:
        findings = [LanguageFinding(
            severity="error",
            category="language",
            message=f"langue: {wrong}/{total} blocs ({round(ratio*100)}%) et attendu '{expected}' — ",
        )]
    else:
        findings = block_warnings
    return LanguageCheckResult(
        expected=expected,
        rejected=rejected,
        findings=findings,
        doc_index=doc_index,
    )
