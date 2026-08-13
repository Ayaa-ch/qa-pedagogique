from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

_CORE = (
    r'(?:[IVXLCDM]+[-.]\d{1,2}[.-][a-z]\)?)'
    r'|(?:[IVXLCDM]+[-.]\d{1,2}\)?)'
    r'|(?:\d{1,2}\.[a-z]\)?)'
    r'|(?:\d{1,2}-[a-z]\)?)'
    r'|(?:[a-z]\))'
    r'|(?:[a-z]\.(?!\w))'
    r'|(?:\d{1,2}\))'
    r'|(?:\d{1,2}\.(?!\d))'
    r'|(?:[IVXLCDM]+\))'
    r'|(?:[IVXLCDM]+\.(?!\w))'
)
_BOLD_RE = re.compile(r'\*\*\s*(' + _CORE + r')\s*\*\*', re.IGNORECASE)
_LINE_START_RE = re.compile(r'^\s*(' + _CORE + r')(?:\s+|$)', re.IGNORECASE)
_ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
_ROMAN_NUMERALS = (
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
)

# ── Abjad (arabe) : translitteration des marqueurs lettres ───────────────
# abjad.yaml donne le sens direct latin -> arabe (a: "أ"). On l'utilise pour
# translitterer les lettres abjad en latin avant la detection existante, puis
# pour restaurer la forme abjadiya a l'ecriture du JSON.

_ABJAD_FROM_LATIN: dict[str, str] = {}
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _ABJAD_DATA = yaml.safe_load(
        (_PROJECT_ROOT / "rules" / "abjad.yaml").read_text(encoding="utf-8")
    )
    _ABJAD_FROM_LATIN = {
        str(k): str(v) for k, v in (_ABJAD_DATA or {}).get("abjad", {}).items()
    }
except Exception:
    pass

_LATIN_FROM_ABJAD: dict[str, str] = {}
for _latin, _arabic in _ABJAD_FROM_LATIN.items():
    _LATIN_FROM_ABJAD[_arabic.replace("ـ", "")] = _latin

_ABJAD_MARKER_RE = (
    re.compile("[" + re.escape("".join(_LATIN_FROM_ABJAD)) + r"](?=[).])")
    if _LATIN_FROM_ABJAD else None
)


def _abjad_to_latin_markers(text: str) -> str:
    """Translitere les lettres abjad en latin, uniquement en position de marqueur
    (lettre suivie de ')' ou '.'), pour reutiliser la detection existante."""
    if not text or _ABJAD_MARKER_RE is None:
        return text
    return _ABJAD_MARKER_RE.sub(
        lambda m: _LATIN_FROM_ABJAD.get(m.group(0), m.group(0)), text
    )


def latin_to_abjad(marker: str) -> str:
    """Restitue les lettres latines de sous-partie d'un label en forme abjad
    (a -> أ, b -> ب). Les romains (I, II...) restent en latin."""
    if not marker or not _ABJAD_FROM_LATIN:
        return marker
    return re.sub(
        r"[a-z]",
        lambda m: _ABJAD_FROM_LATIN.get(m.group(0), m.group(0)),
        marker,
    )


def _extract_markers(text: str) -> list[str]:
    text = _abjad_to_latin_markers(text)
    if not text:
        return []
    found: list[tuple[int, str]] = [(m.start(), m.group(1).strip()) for m in _BOLD_RE.finditer(text)]

    pos = 0
    for line in re.split(r'(<br\s*/?>|\n)', text, flags=re.IGNORECASE):
        if re.fullmatch(r'<br\s*/?>|\n', line or '', re.IGNORECASE):
            pos += len(line)
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("**"):
            m = _LINE_START_RE.match(stripped)
            if m:
                found.append((pos + (len(line) - len(line.lstrip())), m.group(1).strip()))
        pos += len(line)

    found.sort(key=lambda x: x[0])
    seen: set[int] = set()
    return [mk for p, mk in found if not (p in seen or seen.add(p))]

def _roman_to_int(s: str) -> Optional[int]:
    s = s.upper() 
    if not re.fullmatch(r'[IVXLCDM]+', s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total += val if val >= prev else -val
        prev = max(prev, val)
    return total or None

def _int_to_roman(n: int) -> str:
    if n <= 0:
        return str(n)
    out = []
    for val, sym in _ROMAN_NUMERALS:
        while n >= val:
            out.append(sym)
            n -= val
    return ''.join(out)

@dataclass(frozen=True)
class _Marker:
    kind: str                    # letter | digit | roman | roman_num
    value: tuple                 # (n,) ou (roman, num) — identifiant de base
    sub: Optional[int] = None    # lettre de sous-partie (a=0, b=1...) si présente

def _classify_marker(marker: str) -> Optional[_Marker]:
    core = _abjad_to_latin_markers(marker)
    core = core.strip().strip("*").strip()
    core = core[:-1] if core.endswith((")", ".")) else core
    if m := re.fullmatch(r'([IVXLCDMivxlcdm]+)[-.](\d{1,2})\.([a-zA-Z])', core):
        if (rv := _roman_to_int(m.group(1))) is not None:
            return _Marker("roman_num", (rv, int(m.group(2))), ord(m.group(3).lower()) - ord('a'))
    if m := re.fullmatch(r'([IVXLCDMivxlcdm]+)[-.](\d{1,2})', core):
        if (rv := _roman_to_int(m.group(1))) is not None:
            return _Marker("roman_num", (rv, int(m.group(2))), None)
    if m := re.fullmatch(r'(\d{1,2})\.([a-zA-Z])', core):
        return _Marker("digit", (int(m.group(1)),), ord(m.group(2).lower()) - ord('a'))
    if m := re.fullmatch(r'(\d{1,2})-([a-zA-Z])', core):
        return _Marker("digit", (int(m.group(1)),), ord(m.group(2).lower()) - ord('a'))
    if re.fullmatch(r'[a-z]', core):
        return _Marker("letter", (ord(core) - ord('a'),))
    if re.fullmatch(r'\d{1,2}', core):
        return _Marker("digit", (int(core),))
    if re.fullmatch(r'[IVXLCDMivxlcdm]+', core):
        if (rv := _roman_to_int(core)) is not None:
            return _Marker("roman", (rv,))
    return None

def _to_triplet(key: _Marker) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if key.kind == "roman":
        return (key.value[0] - 1, None, None)
    if key.kind == "roman_num":
        return (key.value[0] - 1, key.value[1] - 1, key.sub)
    if key.kind == "digit":
        return (None, key.value[0] - 1, key.sub)
    if key.kind == "letter":
        return (None, None, key.value[0])
    return (None, None, None)

def _canonical_marker(key: _Marker) -> str:
    r, d, l = _to_triplet(key)
    parts = []
    if r is not None:
        parts.append(_int_to_roman(r + 1))
    if d is not None:
        parts.append(str(d + 1))
    out = ".".join(parts)
    if l is not None:
        out += ("." if out else "") + chr(ord('a') + l)
    return out or "?"

def _extract_marker_questions(text: str) -> dict[str, str]:
    text = _abjad_to_latin_markers(text)
    if not text:
        return {}
    
    matches = []
    for m in _BOLD_RE.finditer(text):
        matches.append((m.start(), m.end(), m.group(1).strip()))
        
    pos = 0
    for line in re.split(r'(<br\s*/?>|\n)', text, flags=re.IGNORECASE):
        if re.fullmatch(r'<br\s*/?>|\n', line or '', re.IGNORECASE):
            pos += len(line)
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("**"):
            m = _LINE_START_RE.match(stripped)
            if m:
                start_offset = pos + (len(line) - len(line.lstrip()))
                end_offset = start_offset + m.end()
                matches.append((start_offset, end_offset, m.group(1).strip()))
        pos += len(line)
        
    matches.sort(key=lambda x: x[0])
    
    unique_matches = []
    seen_start = set()
    for start, end, marker in matches:
        if start not in seen_start:
            seen_start.add(start)
            unique_matches.append((start, end, marker))
            
    questions = {}
    for i, (start, end, marker) in enumerate(unique_matches):
        next_start = unique_matches[i+1][0] if i + 1 < len(unique_matches) else len(text)
        q_text = text[end:next_start].strip()
        q_text = re.sub(r'^[ :.,;-]+', '', q_text).strip()
        questions[marker] = q_text
        
    return questions


def build_marker_pairs(
    prompt_text: str,
    correction_labels: list[str],
    correction_texts: list[str],
) -> dict:
    markers = _extract_markers(prompt_text)
    questions = _extract_marker_questions(prompt_text)
    return {
        "marqueurs_questions": [
            {"marqueur": m, "question": questions.get(m, "")} for m in markers
        ],
        "labels_reponses": [
            {"label": lbl, "reponse": txt}
            for lbl, txt in zip(correction_labels, correction_texts)
        ],
    }

# Classification des paires marqueurs <-> labels de correction
GroupKind = Literal["MATCHED", "GLOBAL_VS_SOUSQ", "MISSING", "PARTIAL_MISSING"]

@dataclass
class GroupClassification:
    kind: GroupKind
    exercise_markers: list[str]
    matched_pairs: list[tuple[str, str]] = field(default_factory=list)
    missing_markers: list[str] = field(default_factory=list)
    correction_label: Optional[str] = None

def _base_key(marker: str) -> Optional[tuple]:
    key = _classify_marker(marker)
    return None if key is None else (key.kind, key.value)

def _sub_of(marker: str) -> Optional[int]:
    key = _classify_marker(marker)
    return key.sub if key else None

def _canon(marker: str) -> str:
    key = _classify_marker(marker)
    return _canonical_marker(key) if key else marker

def _group_exercise_markers(markers: list[str]) -> dict[tuple, list[str]]:
    groups: dict[tuple, list[str]] = {}
    for m in markers:
        base = _base_key(m) or ("__unclassified__", m)
        groups.setdefault(base, []).append(m)
    return groups

def _correction_lookup(labels: list[str]) -> tuple[dict[str, str], dict[tuple, str]]:
    exact: dict[str, str] = {}
    global_by_base: dict[tuple, str] = {}
    for lbl in labels:
        if not lbl:
            continue
        key = _classify_marker(lbl)
        if key is None:
            continue
        exact[_canonical_marker(key)] = lbl
        if key.sub is None:
            global_by_base[(key.kind, key.value)] = lbl
    return exact, global_by_base

def classify_marker_pairs(
    exercise_markers: list[str], correction_labels: list[str]
) -> list[GroupClassification]:
    exact, global_by_base = _correction_lookup(correction_labels)
    groups = _group_exercise_markers(exercise_markers)

    results: list[GroupClassification] = []
    for base, ex_markers in groups.items():
        has_subs = any(_sub_of(m) is not None for m in ex_markers)

        matched: list[tuple[str, str]] = []
        missing: list[str] = []
        for m in ex_markers:
            canon = _canon(m)
            if canon in exact:
                matched.append((m, exact[canon]))
            else:
                missing.append(m)

        if not missing:
            results.append(GroupClassification(
                kind="MATCHED", exercise_markers=ex_markers, matched_pairs=matched,
            ))
            continue

        if has_subs and len(missing) == len(ex_markers) and base in global_by_base:
            results.append(GroupClassification(
                kind="GLOBAL_VS_SOUSQ",
                exercise_markers=ex_markers,
                correction_label=global_by_base[base],
            ))
            continue

        if matched:
            results.append(GroupClassification(
                kind="PARTIAL_MISSING", exercise_markers=ex_markers,
                matched_pairs=matched, missing_markers=missing,
            ))
        else:
            results.append(GroupClassification(
                kind="MISSING", exercise_markers=ex_markers, missing_markers=missing,
            ))
    return results


