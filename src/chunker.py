"""Split page text into overlapping chunks that keep their citation metadata.

Why overlap: a sentence that answers the question often straddles a chunk boundary.
Repeating ~150 characters between neighbours means we rarely cut an answer in half.

Why a heading PATH: a chunk deep inside "II. THỰC TRẠNG > 1. THAM GIA BHXH BẮT BUỘC"
says nothing about that on its own. Putting the whole path in front of the text we
embed gives the vector (and the reranker) the context a human reader would have.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import config
from .loader import AMEND_LEVEL, PageBlock, parse_heading

# This corpus is an AMENDING law: its own articles are just "Điều 1. Sửa đổi...".
# What a reader actually needs to cite is the article being amended, which appears
# inside the clause text ("10. Sửa đổi, bổ sung khoản 1 Điều 18 như sau:").
AMENDMENT_PATTERN = re.compile(
    r"^[“\"']?\s*(?:\d+[a-zđ]?\s*[.)]|[a-zđ]\s*\))\s*"
    r"(?:Sửa đổi|Bổ sung|Thay thế|Bãi bỏ)[^\n]{0,140}?(Điều\s+\d+)"
)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?;:])\s+")

TITLE_MAX = 90   # one heading inside a path
PATH_MAX = 300   # the whole path


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int
    heading: str = ""   # the deepest heading: what the citation shows
    path: str = ""      # the full chain of headings: what the embedding sees
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.refresh_id()

    def refresh_id(self) -> None:
        raw = f"{self.source}|{self.page}|{self.index}|{self.text[:80]}"
        self.id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def citation_label(self) -> str:
        parts = [self.source, f"trang {self.page}"]
        if self.heading:
            parts.append(self.heading)
        return " – ".join(parts)

    @property
    def context_label(self) -> str:
        """The heading text put in front of the chunk for embedding / reranking."""
        if config.HEADING_PATH and self.path:
            return self.path
        return self.heading

    @property
    def embedding_text(self) -> str:
        """What we actually send to the embedding model.

        The heading is prepended so the article number becomes part of the vector.
        Without this the system is blind to the very label it displays: a question
        like "Khoản 4 Điều 3 quy định gì" has nothing to match, because the chunk's
        body talks about fees and never mentions which article it belongs to.
        This is the "contextual chunk header" technique.
        """
        label = self.context_label
        if not label:
            return self.text
        return f"{label}\n\n{self.text}"

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "index": self.index,
            "heading": self.heading,
            "path": self.path,
            "citation": self.citation_label,
        }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def _is_table(paragraph: str) -> bool:
    lines = paragraph.split("\n", 2)
    return len(lines) == 3 and lines[0].startswith("| ") and lines[1].startswith("| ---")


def _split_table(table: str, size: int) -> list[str]:
    """Cut a long table into row groups, repeating the header in every piece."""
    if len(table) <= size:
        return [table]
    lines = table.split("\n")
    head, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current = list(head)
    for row in rows:
        if len(current) > 2 and len("\n".join(current + [row])) > size:
            pieces.append("\n".join(current))
            current = list(head)
        current.append(row)
    if len(current) > 2:
        pieces.append("\n".join(current))
    return pieces


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """A paragraph bigger than one chunk gets split on sentence boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_PATTERN.split(paragraph):
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            # A single sentence longer than the chunk size: hard cut it.
            while len(sentence) > size:
                pieces.append(sentence[:size])
                sentence = sentence[size:]
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` characters, started at a word boundary (never mid-word)."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


# --------------------------------------------------------------------------- #
# Headings: a stack, so a chunk knows its whole path
# --------------------------------------------------------------------------- #

Stack = list[tuple[int, str]]


def _heading_of(unit: str) -> tuple[int, str] | None:
    """(level, title) of the heading this paragraph introduces, or None.

    The loader has already isolated headings onto their own paragraphs, so a
    heading is recognised by BEING one — not by containing a matching phrase
    somewhere inside a wall of body text.
    """
    parsed = parse_heading(unit)
    if parsed:
        return parsed[0], parsed[1][:120]

    match = AMENDMENT_PATTERN.match(unit)
    if match:
        return AMEND_LEVEL, f"{match.group(1).strip()} (được sửa đổi)"
    return None


def _push(stack: Stack, level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _label(stack: Stack) -> tuple[str, str]:
    """(deepest heading, full path) for the headings currently in force."""
    if not stack:
        return "", ""
    titles = [t if len(t) <= TITLE_MAX else t[:TITLE_MAX].rstrip() + "…" for _, t in stack]
    path = " > ".join(titles)
    while len(path) > PATH_MAX and len(titles) > 1:  # keep the deepest part
        titles.pop(0)
        path = "… > " + " > ".join(titles)
    return stack[-1][1], path


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_block(block: PageBlock, size: int, overlap: int, start_index: int,
                inherited: Stack | None = None) -> tuple[list[Chunk], Stack]:
    """Turn one page into chunks. Returns the chunks and the heading stack in force
    at the end of the page, so the next page can inherit it (an Điều often spans
    several pages)."""
    paragraphs = [p.strip() for p in block.text.split("\n\n") if p.strip()]

    units: list[tuple[str, bool]] = []  # (text, is_table)
    for paragraph in paragraphs:
        if _is_table(paragraph):
            units.extend((piece, True) for piece in _split_table(paragraph, size))
        elif len(paragraph) <= size:
            units.append((paragraph, False))
        else:
            # Continuation pieces carry no marker, so they inherit the heading.
            units.extend((piece, False) for piece in _split_long_paragraph(paragraph, size))

    chunks: list[Chunk] = []
    stack: Stack = list(inherited or [])
    chunk_label = _label(stack)  # headings in force where THIS chunk starts
    buffer = ""
    has_table = False
    index = start_index

    def flush(buf: str, label: tuple[str, str], carry_tail: bool) -> str:
        nonlocal index
        if not buf.strip():
            return ""
        chunks.append(
            Chunk(text=buf.strip(), source=block.source, page=block.page, index=index,
                  heading=label[0], path=label[1])
        )
        index += 1
        return _tail(buf, overlap) if carry_tail else ""

    for unit, is_table in units:
        if not is_table:
            found = _heading_of(unit)
            if found:
                _push(stack, *found)

        if not buffer:
            # New chunk begins here: label it with the heading in force NOW,
            # not with whatever article happens to start later inside it.
            chunk_label = _label(stack)
            buffer, has_table = unit, is_table
        elif len(buffer) + len(unit) + 2 <= size:
            buffer = f"{buffer}\n\n{unit}"
            has_table = has_table or is_table
        else:
            # No overlap next to a table: half a row is worse than no overlap.
            tail = flush(buffer, chunk_label, carry_tail=not (has_table or is_table))
            chunk_label = _label(stack)
            buffer = f"{tail}\n\n{unit}".strip() if tail else unit
            has_table = is_table

    flush(buffer, chunk_label, carry_tail=False)
    return chunks, stack


def _is_noise(chunk: Chunk, min_chars: int) -> bool:
    """Chunks that carry no answer: a lone page number, a cover line, headings only."""
    text = chunk.text.strip()
    if len(text) < min_chars:
        return True
    if sum(c.isalpha() for c in text) < 20:
        return True
    return all(parse_heading(p) is not None for p in text.split("\n\n"))


def chunk_blocks(blocks: list[PageBlock], size: int, overlap: int,
                 min_chars: int | None = None) -> list[Chunk]:
    """Chunk every block. Indices restart at 0 for each source file, so a file
    that is uploaded later gets the same indices as when the whole folder is
    ingested (neighbour look-ups rely on index +/- 1 within one source)."""
    min_chars = config.MIN_CHUNK_CHARS if min_chars is None else min_chars
    stack_by_source: dict[str, Stack] = {}
    kept: list[Chunk] = []
    counters: dict[str, int] = {}

    for block in blocks:
        chunks, stack_by_source[block.source] = chunk_block(
            block, size, overlap, start_index=0,
            inherited=stack_by_source.get(block.source),
        )
        for chunk in chunks:
            if _is_noise(chunk, min_chars):
                continue
            chunk.index = counters.get(chunk.source, 0)
            counters[chunk.source] = chunk.index + 1
            chunk.refresh_id()
            kept.append(chunk)

    return kept
