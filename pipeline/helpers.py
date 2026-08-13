import json
import re
import subprocess
import sys
import time
from typing import Optional

import requests

from parser import Block, Section, Slide


# ── Parsing de JSON renvoye par un LLM ──────────────────────────────────

def parse_llm_json(raw: str) -> Optional[dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


# ── Serveur Ollama ───────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_alive(url: str = OLLAMA_BASE_URL, timeout: float = 3.0) -> bool:
    try:
        requests.get(f"{url}/api/tags", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def ensure_ollama_server(url: str = OLLAMA_BASE_URL, timeout: float = 60.0) -> None:
    """S'assure que le serveur Ollama repond, en lançant 'ollama serve' sinon."""
    if _ollama_alive(url):
        return
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except OSError as exc:
        raise RuntimeError(
            "commande 'ollama' introuvable — installez Ollama ou lancez 'ollama serve'"
        ) from exc
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _ollama_alive(url):
            return
        time.sleep(0.5)
    raise RuntimeError("serveur Ollama injoignable apres lancement de 'ollama serve'")


def block_texts(block: Block) -> list[str]:
    d = block.data
    return [t for t in [d.text, d.prompt, d.method, d.error, d.tip] if t]

# ── Helpers de netoyage de texte ────────────────────────────────────────
def strip_markup(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"^\|.*\|$", " ", text, flags=re.MULTILINE)
    return re.sub(r"\s{2,}", " ", text).strip()

# ── Helpers d'extraction de texte ────────────────────────────────────────

def slide_resume_text(slide: Slide) -> str:
    parts = []
    for b in slide.blocks:
        if b.type == "text" and b.data.text:
            parts.append(b.data.text)
    return " ".join(parts)

def resume_slide_texts(sections: list[Section]) -> dict[int, str]:
    out: dict[int, str] = {}
    for sec in sections:
        if sec.kind != "resume":
            continue
        for slide in sec.slides:
            out[slide.id] = slide_resume_text(slide)
    return out

def exercise_text(slide: Slide) -> str:
    texts = [b.data.text for b in slide.blocks if b.type == "text" and b.data.text]
    prompts = [b.data.prompt for b in slide.blocks if b.type == "question" and b.data.prompt]
    return "\n\n".join(texts + prompts)


# ── Construction de texte depuis une slide ───────────────────────────────

def prompt_text(exercise_slide: Slide) -> str:
    return "\n\n".join(
        b.data.prompt for b in exercise_slide.blocks
        if b.type == "question" and b.data.prompt
    )


def enonce_text(exercise_slide: Slide) -> str:
    return "\n\n".join(
        (b.data.text or "").strip()
        for b in exercise_slide.blocks
        if b.type == "text" and (b.data.text or "").strip()
    )


# ── Nettoyage de texte ───────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()
