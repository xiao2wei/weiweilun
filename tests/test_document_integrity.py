from __future__ import annotations

import re
from pathlib import Path

import pytest


PAPER_FILES = ("方案1.md", "论文1.md")

# These fragments are typical results of reading LaTeX through a language
# string literal and then writing the interpreted value back to Markdown.
BROKEN_LATEX_PATTERNS = (
    (re.compile(r"(?<![A-Za-z\\])rac(?=(?:\d|\{))"), "bare frac fragment"),
    (re.compile(r"\\(?:lefts|righte)\b"), "damaged delimiter command"),
    (re.compile(r"\t(?:au|ext|heta|imes)(?![A-Za-z])"), "tab-damaged command"),
    (re.compile(r"(?:^|\n)(?:abla|u)(?=[_^{\\])"), "newline-damaged command"),
)


@pytest.mark.parametrize("relative_path", PAPER_FILES)
def test_research_markdown_is_strict_utf8_and_latex_is_not_escape_damaged(
    repo_root: Path, relative_path: str
) -> None:
    path = repo_root / relative_path
    assert path.is_file(), f"required research document is missing: {relative_path}"
    raw = path.read_bytes()
    assert raw, f"required research document is empty: {relative_path}"
    text = raw.decode("utf-8", errors="strict")

    abnormal_controls = [
        (index, f"U+{ord(character):04X}")
        for index, character in enumerate(text)
        if (ord(character) < 32 and character not in "\t\n\r") or ord(character) == 127
    ]
    assert not abnormal_controls, (
        f"{relative_path} contains abnormal control characters: "
        f"{abnormal_controls[:10]}"
    )
    bare_carriage_returns = [
        index
        for index, character in enumerate(text)
        if character == "\r" and (index + 1 == len(text) or text[index + 1] != "\n")
    ]
    assert not bare_carriage_returns, (
        f"{relative_path} contains carriage returns that are not CRLF newlines: "
        f"{bare_carriage_returns[:10]}"
    )
    assert "\ufffd" not in text, f"{relative_path} contains Unicode replacement text"

    damaged = [
        (label, match.start(), match.group(0))
        for pattern, label in BROKEN_LATEX_PATTERNS
        for match in pattern.finditer(text)
    ]
    assert not damaged, (
        f"{relative_path} contains common damaged LaTeX escapes: {damaged[:10]}"
    )
