import re
from dataclasses import dataclass
from typing import Literal, Optional

from parser import Document
from helpers import block_texts


@dataclass
class RenderFinding:
    severity: Literal["error", "warning"]
    category: str
    message: str
    block_id: Optional[int] = None
    slide_id: Optional[int] = None

    def __str__(self) -> str:
        prefix = "[ERROR]" if self.severity == "error" else "[WARNING]"
        return f"{prefix} {self.message} "


def _find_unbalanced_command(text: str, command: str) -> bool:
    for match in re.finditer(rf"\\{command}\{{", text):
        depth = 1
        pos = match.end()
        while pos < len(text):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        if depth != 0:
            return True
    return False

def _check_latex_environments(text: str) -> list[str]:
    """Retourne  \\begin{...} qui n'ont pas de \\end{...} """
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([a-zA-Z0-9\*]+)\}", text):
        kind, env = match.group(1), match.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] == env:
                    del stack[i:]
                    break
    return list(reversed(stack))


_STRAY_NEWLINE_RE = re.compile(r"\\n(?![a-zA-Z])")
def _check_stray_newlines(text: str, block_id: int, slide_id: int) -> list[RenderFinding]:
    """
    Detecte les sequences litterales "\\n" 
    """
    problems = []
    count = len(_STRAY_NEWLINE_RE.findall(text))
    if count:
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: {count} sequence(s) \\n litterale(s) "
            f"(echappement en trop, devrait etre un vrai saut de ligne)",
            block_id, slide_id,
        ))
    return problems


_MATH_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
def _check_newlines_in_math(text: str, block_id: int, slide_id: int) -> list[RenderFinding]:
    """
    Detecte un VRAI caractere de saut de ligne a l'interieur d'un bloc $$...$$.
    """
    problems = []
    for match in _MATH_BLOCK_RE.finditer(text):
        if "\n" in match.group(1) or "\r" in match.group(1):
            problems.append(RenderFinding(
                "warning", "latex",
                f"block {block_id} slide {slide_id}: saut de ligne reel dans un bloc $$...$$ "
                f"(peut s'afficher comme du texte litteral)",
                block_id, slide_id,
            ))
            break
    return problems

def _check_latex(text: str, block_id: int, slide_id: int) -> list[RenderFinding]:
    problems = []

    if text.count("$$") % 2 != 0:
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: $$ non fermé",
            block_id, slide_id,
        ))

    inline_text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    if inline_text.count("$") % 2 != 0:
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: $ inline non fermé",
            block_id, slide_id,
        ))

    unclosed_envs = _check_latex_environments(text)
    for env in unclosed_envs:
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: \\begin{{{env}}} non fermé par \\end{{{env}}}",
            block_id, slide_id,
        ))

    if text.count("{") != text.count("}"):
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: accolades non équilibrées",
            block_id, slide_id,
        ))

    if _find_unbalanced_command(text, "text"):
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: \\text{{ non fermé",
            block_id, slide_id,
        ))

    if _find_unbalanced_command(text, "frac"):
        problems.append(RenderFinding(
            "error", "latex",
            f"block {block_id} slide {slide_id}: \\frac{{ non fermé",
            block_id, slide_id,
        ))

    problems.extend(_check_stray_newlines(text, block_id, slide_id))
    problems.extend(_check_newlines_in_math(text, block_id, slide_id))

    return problems


def _check_bold_spans_in_math(text: str, block_id: int, slide_id: int) -> list[RenderFinding]:
    """
    Detecte les spans **...** qui englobent un bloc $$...$$.
    """
    TOKEN_RE = re.compile(r"(\*\*|\$\$.*?\$\$)", re.DOTALL)
    open_bold = False
    for tok in TOKEN_RE.split(text):
        if tok == "**":
            open_bold = not open_bold
        elif tok.startswith("$$") and open_bold:
            return [RenderFinding(
                "warning", "markdown",
                f"block {block_id} slide {slide_id}: span **...** traverse un bloc $$...$$ "
                f"(le ** fermant sera orphelin a l'ecran)",
                block_id, slide_id,
            )]
    return []

def _check_markdown(text: str, block_id: int, slide_id: int) -> list[RenderFinding]:
    problems = []

    if text.count("**") % 2 != 0:
        problems.append(RenderFinding(
            "warning", "markdown",
            f"block {block_id} slide {slide_id}: ** non fermé",
            block_id, slide_id,
        ))

    problems.extend(_check_bold_spans_in_math(text, block_id, slide_id))

    normalized = re.sub(r"<br\s*/?>", "\n", text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if table_lines:
        if len(table_lines) < 3 or not re.match(r"^\|[-:|\s]+\|$", table_lines[1]):
            problems.append(RenderFinding(
                "warning", "markdown",
                f"block {block_id} slide {slide_id}: tableau Markdown mal formé",
                block_id, slide_id,
            ))
        elif not any("|" in line for line in table_lines[2:]):
            problems.append(RenderFinding(
                "warning", "markdown",
                f"block {block_id} slide {slide_id}: tableau Markdown sans ligne de données",
                block_id, slide_id,
            ))

    return problems

def check_render(documents: list[Document]) -> tuple[list[RenderFinding], list[RenderFinding]]:
    errors, warnings = [], []

    for doc in documents:
        for sec in doc.content.sections:
            for slide in sec.slides:
                for block in slide.blocks:
                    for raw in block_texts(block):
                        findings = (
                            _check_latex(raw, block.id, slide.id)
                            + _check_markdown(raw, block.id, slide.id)
                        )
                        for f in findings:
                            (errors if f.severity == "error" else warnings).append(f)

    return errors, warnings