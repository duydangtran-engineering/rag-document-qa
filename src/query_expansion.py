"""Expand abbreviations in the question before searching.

Why: the embedding model has never learned that "LĐPCT" means
"lao động phi chính thức". The eval question "Tỷ lệ LĐPCT ở nông thôn chiếm bao
nhiêu?" therefore ranked the right chunk 13th, because that chunk spells the
term out in full. Writing both forms into the query fixes that:

    "Tỷ lệ LĐPCT ở nông thôn"  ->  "Tỷ lệ LĐPCT (lao động phi chính thức) ở nông thôn"

The expanded text is used for search and reranking only; the model answering
the question still sees exactly what the user typed.

Add your own abbreviations in data/abbreviations.json, e.g.
    {"PCCC": "phòng cháy chữa cháy", "CBCNV": "cán bộ công nhân viên"}
Entries there override the built-in list below.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from . import config

# Only abbreviations that are unambiguous in the documents we have. A wrong
# expansion hurts more than a missing one, so keep this list conservative.
BUILT_IN: dict[str, str] = {
    # Bảo hiểm / lao động
    "BHXH": "bảo hiểm xã hội",
    "BHXHTN": "bảo hiểm xã hội tự nguyện",
    "BHXHBB": "bảo hiểm xã hội bắt buộc",
    "BHYT": "bảo hiểm y tế",
    "BHTN": "bảo hiểm thất nghiệp",
    "ASXH": "an sinh xã hội",
    "TGXH": "trợ giúp xã hội",
    "LĐPCT": "lao động phi chính thức",
    "LĐCT": "lao động chính thức",
    "LLLĐ": "lực lượng lao động",
    "NLĐ": "người lao động",
    "NSDLĐ": "người sử dụng lao động",
    "HĐLĐ": "hợp đồng lao động",
    # Hành chính / pháp luật
    "UBND": "Ủy ban nhân dân",
    "HĐND": "Hội đồng nhân dân",
    # Xử lý ảnh
    "NMS": "Non-Maximum Suppression",
    "IoU": "Intersection over Union, giao trên hợp",
}


@lru_cache(maxsize=1)
def _table() -> tuple[dict[str, str], re.Pattern | None]:
    table = dict(BUILT_IN)
    try:
        extra = json.loads(config.ABBREVIATIONS_FILE.read_text(encoding="utf-8"))
        table.update({str(k): str(v) for k, v in extra.items()})
    except FileNotFoundError:
        pass
    except Exception as exc:  # A broken file must not take the search down.
        print(f"[query_expansion] Bỏ qua {config.ABBREVIATIONS_FILE.name}: {exc}")

    if not table:
        return table, None
    # Longest first, so "BHXHTN" wins over "BHXH". Lookarounds instead of \b
    # because \b does not treat "Đ" consistently as part of a word.
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE
    )
    return {k.lower(): v for k, v in table.items()}, pattern


def expand(question: str) -> str:
    table, pattern = _table()
    if pattern is None:
        return question

    def replace(match: re.Match) -> str:
        abbr = match.group(1)
        full = table[abbr.lower()]
        # Already spelled out right after it ("LĐPCT (lao động phi chính thức)")?
        tail = question[match.end():match.end() + len(full) + 3].lower()
        if full.lower() in tail:
            return abbr
        return f"{abbr} ({full})"

    return pattern.sub(replace, question)
