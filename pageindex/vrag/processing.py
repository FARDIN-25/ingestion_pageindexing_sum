from __future__ import annotations
"""Text processing, deduplication, chunking, and compression utilities for vectorless RAG."""
from .schema import LEVEL

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    t = _WS.sub(" ", (text or "").strip().lower())
    t = re.sub(r"[^\w\s]", "", t)
    return t


def sha256_content(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", normalize_for_hash(text)))


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_synthetic_title(text: str) -> bool:
    """LLM paraphrase titles — comma lists, engagement boilerplate, etc."""
    t = (text or "").strip()
    if not t:
        return True
    if re.match(r"^(?:SECTION|UNIT|\d+\.\d+)", t, re.I):
        return False
    if t.count(",") >= 2 and len(t) > 40:
        return True
    synthetic_kw = (
        "engagement terms",
        "reporting standards",
        "materiality",
        "acceptance",
        "terms,",
        "standards,",
        "overview of",
        "introduction to",
    )
    low = t.lower()
    if any(k in low for k in synthetic_kw) and not re.match(r"^\d+\.\d+", t):
        if len(t) > 35 or t.count(",") >= 1:
            return True
    if len(t) > 50 and not re.match(r"^\d+\.\d+(?:\.\d+)?\s+", t):
        if t.count(".") >= 1 and re.search(r"\b(and|or|including)\b", low):
            return True
    return False


RE_LEGAL_SECTION_REF = re.compile(
    r"^Section\s+\d{2,}[A-Z]{0,4}(?:\s+[A-Z]{1,4})?\s+(?:of\s+the\s+Act|under\s+the\s+Act|of\s+the|or\s+section|or\s+under|and\s+section)\b",
    re.I,
)
RE_LEGAL_SECTION_LABEL = re.compile(r"^Section\s+\d{2,}[A-Z]{0,4}(?:\s+[A-Z]{1,4})?\s*:\s*", re.I)
RE_LEGAL_SECTION_CHAIN = re.compile(r"\bor\s+section\s+\d", re.I)


def is_legal_section_reference_title(text: str) -> bool:
    """Tax/act prose citations masquerading as headings (e.g. 'Section 44A B of the Act.')."""
    t = (text or "").strip()
    if not t:
        return False
    if RE_LEGAL_SECTION_REF.match(t):
        return True
    if RE_LEGAL_SECTION_LABEL.match(t):
        return True
    if RE_LEGAL_SECTION_CHAIN.search(t):
        return True
    if re.match(r"^Section\s+\d{2,}[A-Z]{0,4}\b", t, re.I) and (
        " of the " in t.lower() or t.rstrip().endswith(":") or len(t) > 45
    ):
        if not re.match(r"^Section\s+\d+\.\d+", t):
            return True
    return False


def is_paragraph_title(text: str) -> bool:
    """True if text looks like body prose, not a structural heading."""
    t = (text or "").strip()
    if not t:
        return True
    if is_legal_section_reference_title(t):
        return True
    if is_synthetic_title(t):
        return True
    if re.match(r"^(?:provisions|introduction|background|definitions)\s+(?:of|to|for|under|in)\b", t, re.I):
        return True
    if re.match(r"^(?:SECTION|UNIT|CHAPTER)\s+", t, re.I):
        return False
    if re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", t) and len(t) <= 90:
        return False
    if re.match(r"^(?:FORM\s+)?GST\s", t, re.I) and len(t) <= 80:
        return False
    if RE_BOOK_SECTION.match(t) and len(t) <= 90:
        return False
    first_letter = re.search(r"[a-zA-Z]", t)
    if first_letter and first_letter.group(0).islower() and not re.match(r"^\d+\.\d+", t):
        return True
    if t.endswith(".") and not re.match(r"^\d+\.\d+", t):
        return True
    if re.search(r"\b(or|and|of|the|to|in|under|with|for|shall|has|been|which|is|by|its|on|from)\s+section\b", t, re.I):
        return True
    if t.endswith(":") and not re.match(r"^(?:SECTION|UNIT|CHAPTER|FORM)\b", t, re.I):
        if len(t.split()) > 3:
            return True
    if len(t) > 55:
        return True
    if re.search(r"\b(the|for the|according to|shall be|has been|which is|except where|furnished after|commencement of)\b", t, re.I):
        return True
    if t.count(".") >= 2 or t.count(",") >= 2:
        return True
    return False


def is_table_of_contents_content(title: str, raw: str) -> bool:
    """Detect TOC pages from title and dotted-leader line patterns."""
    tl = (title or "").lower().strip()
    if re.search(r"\b(table\s+of\s+contents|^\s*contents\s*$)\b", tl):
        return True
    lines = [ln.strip() for ln in (raw or "").split("\n") if ln.strip()][:50]
    if len(lines) < 4:
        return False
    leader_hits = 0
    for ln in lines:
        if re.search(r"\.{3,}\s*\d{1,4}\s*$", ln):
            leader_hits += 1
        elif re.search(r"\s{2,}\d{1,3}\s*$", ln) and len(ln) < 120:
            leader_hits += 1
    return leader_hits >= max(3, int(len(lines) * 0.3))


"""Text sanitation: OCR cleanup, normalization, garbage removal."""

import re
import unicodedata

GARBAGE_PATTERNS = [
    re.compile(r"\(Detected as", re.I),
    re.compile(r"^This (instruction|document) asks", re.I),
    re.compile(r"^The response should", re.I),
    re.compile(r"^Reply format", re.I),
    re.compile(r"^Directly return", re.I),
    re.compile(r"^concise summary", re.I),
    re.compile(r"^without any introductory", re.I),
    re.compile(r"^Please note:", re.I),
    re.compile(r"^system:\s*", re.I),
    re.compile(r"^assistant:\s*", re.I),
    re.compile(r"^user:\s*", re.I),
    re.compile(r"^OCR\s*(metadata|confidence)", re.I),
    re.compile(r"^extraction\s+instruction", re.I),
    re.compile(r"^parser\s+label", re.I),
    re.compile(r"^internal\s+pipeline", re.I),
]

PHYSICAL_INDEX = re.compile(r"<\s*physical_index_\d+\s*>", re.I)
PHYSICAL_INDEX_LINE = re.compile(r"^<\s*physical_index_\d+\s*>$", re.I)

PAGE_NUM_ONLY = re.compile(r"^\d{1,3}$")
PAGE_MARKER = re.compile(r"^(?:page\s+)?\d{1,3}\s*(?:of\s+\d{1,3})?$", re.I)
HEADER_FOOTER = re.compile(
    r"^(?:page\s+\d+|www\.|http://|https://|all copyrights|self-instructional study material|"
    r"reprint\s+\d{4}|jagat guru|panjab university|established by act)$",
    re.I,
)
TABLE_JUNK = re.compile(r"^\d{5,}\.\d{2}\s+\d{1,2}-[A-Za-z]+")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_physical_index(text: str) -> str:
    if not text:
        return ""
    text = PHYSICAL_INDEX.sub("", text)
    lines = []
    for ln in text.split("\n"):
        if PHYSICAL_INDEX_LINE.match(ln.strip()):
            continue
        lines.append(ln)
    return "\n".join(lines)


def is_garbage_line(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 2:
        return True
    if PHYSICAL_INDEX_LINE.match(t) or PHYSICAL_INDEX.search(t):
        return True
    if PAGE_NUM_ONLY.match(t) or PAGE_MARKER.match(t):
        return True
    if HEADER_FOOTER.match(t):
        return True
    if TABLE_JUNK.match(t) and len(t) > 40:
        return True
    return any(p.search(t) for p in GARBAGE_PATTERNS)


def contains_garbage_artifact(text: str) -> bool:
    low = (text or "").lower()
    markers = (
        "detected as legal book",
        "detected as legal act",
        "ocr metadata",
        "extraction instruction",
        "parser label",
        "internal pipeline",
        "this instruction asks",
        "physical_index",
        "<physical_index",
    )
    return any(m in low for m in markers) or bool(PHYSICAL_INDEX.search(text or ""))


def sanitize_line(text: str) -> str:
    t = normalize_unicode(text)
    t = strip_physical_index(t)
    t = t.replace("\u00ad", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\(Detected as[^)]*\)", "", t, flags=re.I).strip()
    t = re.sub(r"^[-–—]\s*", "", t)
    return t


def join_hyphenated_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if buf.endswith("-") and len(buf) > 1:
            buf = buf[:-1] + line
        elif buf:
            out.append(buf)
            buf = line
        else:
            buf = line
    if buf:
        out.append(buf)
    return out


def sanitize_raw_content(text: str) -> str:
    if not text:
        return ""
    text = strip_physical_index(text)
    lines = [sanitize_line(ln) for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not is_garbage_line(ln)]
    lines = join_hyphenated_lines(lines)
    return "\n".join(lines).strip()


"""Deterministic heading detection — structural titles only."""

import re
from typing import Optional





RE_SECTION = re.compile(r"^SECTION\s+([A-Z])\s*$", re.I)
RE_UNIT = re.compile(r"^UNIT\s*(?:NO:?\s*)?([IVXLC\d]+)\s*[:\.]?\s*(.*)$", re.I)
RE_UNIT_SHORT = re.compile(r"^UNIT\s+([IVXLC\d]+)\s*$", re.I)
RE_TOPIC = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")
RE_SUBTOPIC = re.compile(r"^(\d+)\.(\d+)\.(\d+)\s+(.+)$")
RE_PREFACE = re.compile(r"^PREFACE\s*$", re.I)
RE_OBJECTIVES = re.compile(r"^(?:OBJECTIVE|OBJECTIVES)\s*:?\s*$", re.I)
RE_REFERENCES = re.compile(r"^(?:SUGGESTED\s+)?READINGS?\s*:?\s*$", re.I)
RE_COURSE = re.compile(r"^COURSE\s*(?:COORDINATOR|INFO)?", re.I)
RE_SYLLABUS = re.compile(r"^(?:SYLLABUS|MAX\.\s*MARKS|INSTRUCTIONS\s+FOR|CREDITS\s*:)", re.I)
RE_STRUCTURE = re.compile(r"^STRUCTURE\s*$", re.I)
RE_GST_FORM = re.compile(
    r"^(?:\d+\.\d+\s+)?(?:FORM\s+)?GST\s+(?:REG|GSTR|CMP|PMT|PMT-?\d|DRC)",
    re.I,
)
RE_GST_TOPIC = re.compile(
    r"^(\d+)\.(\d+)\s+(?:FORM\s+)?(?:GST|GSTR|DRC|CMP|PMT)",
    re.I,
)
RE_LET_SUM = re.compile(r"^\d+\.\d+\s+(?:Let\s+us\s+sum\s+up|Let\s+Sum\s+Up|Test\s+Your\s+Knowledge)", re.I)
RE_BOOK_SECTION = re.compile(
    r"^(?:INTRODUCTION|BACKGROUND|DEFINITIONS?|PROVISIONS?|SUMMARY|CONCLUSION|"
    r"APPENDIX|GLOSSARY|PREFACE|ACKNOWLEDGEMENTS?|TABLE\s+OF\s+CONTENTS|CONTENTS|INDEX|FOREWORD)\b",
    re.I,
)
RE_CHAPTER = re.compile(r"^(?:CHAPTER|UNIT|PART)\s+([IVXLC\d]+)", re.I)

FRONT_KW = (
    "preface", "course coordinator", "objectives", "syllabus",
    "suggested reading", "max. marks", "instructions for",
)


def clean_title(raw: str, node_type: str) -> str:
    t = strip_physical_index(raw)
    t = re.sub(r"\s+", " ", t.strip())
    t = re.sub(r"\(Detected as[^)]*\)", "", t, flags=re.I).strip()
    # Normalize OCR artifacts
    t = re.sub(r'Section\s+(\d+[A-Z])\s+([A-Z])\b', r'Section \1\2', t)
    t = re.sub(r'Section\s+(\d+[a-z])\s+([a-z])\b', r'Section \1\2', t)
    t = re.sub(r'([A-Z]{2,})\s+(-)\s+([A-Z])', r'\1-\3', t)
    
    if is_paragraph_title(t) and node_type in ("TOPIC", "SUBTOPIC", "CONTENT"):
        m = RE_TOPIC.match(t) or RE_SUBTOPIC.match(t)
        if m:
            parts = m.groups()
            if len(parts) >= 3 and parts[-1]:
                return f"{parts[0]}.{parts[1]} {parts[-1][:60]}".strip()
        return t[:60].strip() if len(t) > 60 else t
    if len(t) > 90:
        m = RE_TOPIC.match(t) or RE_SUBTOPIC.match(t)
        if m:
            return t[:90].strip()
    return t


def is_structural_heading_line(raw: str) -> bool:
    raw = raw.strip()
    if not raw or len(raw) > 100:
        return False
    if is_paragraph_title(raw) or is_legal_section_reference_title(raw):
        return False
    
    m_sub = RE_SUBTOPIC.match(raw)
    if m_sub:
        return True
        
    m_top = RE_TOPIC.match(raw)
    if m_top:
        title_part = m_top.group(3)
        if re.match(r"^[-+0-9.,\s]+$", title_part) or re.match(r"^\d+\.\d+", title_part):
            return False
        return True

    if RE_SECTION.match(raw) or RE_UNIT.match(raw) or RE_UNIT_SHORT.match(raw):
        return True
    if RE_GST_FORM.match(raw) or RE_GST_TOPIC.match(raw):
        return True
    if RE_LET_SUM.match(raw):
        return True
    if RE_PREFACE.match(raw) or RE_OBJECTIVES.match(raw) or RE_REFERENCES.match(raw):
        return True
    if RE_COURSE.match(raw) and len(raw) < 90:
        return True
    if RE_SYLLABUS.match(raw):
        return True
    if RE_BOOK_SECTION.match(raw) and len(raw) < 90:
        return True
    if RE_CHAPTER.match(raw):
        return True
    return False


def is_valid_heading_title(text: str) -> bool:
    """Structural heading suitable for node title."""
    t = (text or "").strip()
    if not t or len(t) > 100:
        return False
    if is_paragraph_title(t) or is_legal_section_reference_title(t):
        return False
    return is_structural_heading_line(t) or (
        RE_BOOK_SECTION.match(t) and len(t) <= 90
    )


def classify_heading(
    line: DocLine,
    median_font: float,
    body_started: bool,
    current_section: Optional[str],
) -> Optional[tuple[str, str, int, dict]]:
    raw = line.text.strip()
    if not raw or len(raw) > 120:
        return None

    # Check for typographic heading (bold or larger font, short, not a paragraph)
    is_typo_heading = False
    if (line.is_bold or line.font_size > median_font + 0.5) and len(raw.split()) < 12:
        if not is_paragraph_title(raw) and not is_legal_section_reference_title(raw):
            is_typo_heading = True

    if not is_structural_heading_line(raw) and not is_typo_heading and not (
        not body_started and any(k in raw.lower() for k in FRONT_KW) and len(raw) < 100
    ):
        return None

    title = clean_title(raw, "TOPIC")

    if is_typo_heading and not is_structural_heading_line(raw):
        if not body_started:
            return ("PREFACE", title, LEVEL["PREFACE"], {})
        return ("TOPIC", title, LEVEL["TOPIC"], {})

    m = RE_SUBTOPIC.match(raw)
    if m:
        return ("SUBTOPIC", title, LEVEL["SUBTOPIC"], {
            "topic": f"{m.group(1)}.{m.group(2)}.{m.group(3)}",
        })

    m = RE_TOPIC.match(raw)
    if m:
        title_part = m.group(3)
        if re.match(r"^[-+0-9.,\s]+$", title_part) or re.match(r"^\d+\.\d+", title_part):
            pass
        else:
            ntype = "SUBTOPIC" if RE_SUBTOPIC.match(raw) else "TOPIC"
            return (ntype, title, LEVEL.get(ntype, LEVEL["TOPIC"]), {
                "topic": f"{m.group(1)}.{m.group(2)}",
            })

    m = RE_SECTION.match(raw)
    if m:
        return ("SECTION", title, LEVEL["SECTION"], {"section": m.group(1).upper()})

    m = RE_UNIT.match(raw) or RE_UNIT_SHORT.match(raw)
    if m:
        if not body_started:
            return None
        return ("UNIT", title, LEVEL["UNIT"], {"unit": m.group(1).upper()})

    m = RE_CHAPTER.match(raw)
    if m:
        return ("CHAPTER", title, LEVEL.get("CHAPTER", 2), {"chapter": m.group(1).upper()})

    if RE_STRUCTURE.match(raw):
        return None

    if RE_PREFACE.match(raw):
        return ("PREFACE", title, LEVEL["PREFACE"], {})
    if RE_OBJECTIVES.match(raw):
        return ("OBJECTIVES", title, LEVEL["OBJECTIVES"], {})
    if RE_REFERENCES.match(raw):
        return ("REFERENCES", title, LEVEL["REFERENCES"], {})
    if RE_COURSE.match(raw) and len(raw) < 90:
        return ("COURSE_INFO", title, LEVEL["COURSE_INFO"], {})
    if RE_SYLLABUS.match(raw):
        return ("SYLLABUS", title, LEVEL["SYLLABUS"], {})

    if RE_BOOK_SECTION.match(raw):
        low = raw.lower()
        if "contents" in low or "table of" in low:
            return ("TABLE_OF_CONTENTS", title, LEVEL.get("TABLE_OF_CONTENTS", 3), {})
        if any(k in low for k in ("preface", "foreword", "acknowledg")):
            return ("PREFACE", title, LEVEL["PREFACE"], {})
        if "glossary" in low or "index" in low:
            return ("REFERENCES", title, LEVEL["REFERENCES"], {})
        if "introduction" in low or "background" in low or "definition" in low:
            return ("TOPIC", title, LEVEL["TOPIC"], {"section": "intro"})
        return ("TOPIC", title, LEVEL["TOPIC"], {})

    if RE_CHAPTER.match(raw):
        m = RE_CHAPTER.match(raw)
        return ("UNIT", title, LEVEL["UNIT"], {"unit": m.group(1).upper()})

    if body_started and (RE_GST_FORM.match(raw) or RE_GST_TOPIC.match(raw) or RE_LET_SUM.match(raw)):
        return ("TOPIC", title, LEVEL["TOPIC"], {})

    if not body_started:
        low = raw.lower()
        if "preface" in low:
            return ("PREFACE", title, LEVEL["PREFACE"], {})
        if "objective" in low:
            return ("OBJECTIVES", title, LEVEL["OBJECTIVES"], {})
        if "reading" in low:
            return ("REFERENCES", title, LEVEL["REFERENCES"], {})
        if any(k in low for k in ("syllabus", "max. marks", "credits")):
            return ("SYLLABUS", title, LEVEL["SYLLABUS"], {})

    return None


def detect_toc_pages(doc_lines: list[DocLine]) -> set[int]:
    page_texts = {}
    for ln in doc_lines:
        page_texts.setdefault(ln.page, []).append(ln.text)
    toc_pages = set()
    for pnum, lines in page_texts.items():
        if is_table_of_contents_content("", "\n".join(lines)):
            toc_pages.add(pnum)
    return toc_pages


def detect_headings(doc_lines: list[DocLine]) -> list[dict]:
    sizes = [ln.font_size for ln in doc_lines if ln.font_size > 0]
    median = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

    toc_pages = detect_toc_pages(doc_lines)

    headings: list[dict] = []
    body_started = False
    current_section: Optional[str] = None

    for i, ln in enumerate(doc_lines):
        hit = classify_heading(ln, median, body_started, current_section)
        if not hit:
            continue
        ntype, title, level, meta = hit
        
        # If it's a TOC page, skip all TOPIC/SUBTOPIC/PREFACE/OBJECTIVES/etc. headings
        if ln.page in toc_pages and ntype in ("TOPIC", "SUBTOPIC", "PREFACE", "OBJECTIVES", "REFERENCES", "SYLLABUS", "COURSE_INFO"):
            continue
            
        if ntype in ("CHAPTER", "APPENDIX", "SECTION", "UNIT"):
            body_started = True
            if ntype == "SECTION":
                current_section = meta.get("section")
        elif ntype in ("TOPIC", "SUBTOPIC") and body_started:
            pass
            
        headings.append({
            "line_idx": i,
            "type": ntype,
            "title": title,
            "level": level,
            "meta": meta,
            "page": ln.page,
        })

    # Filter out TOC/Outline headings that are clustered together on the same page
    filtered_headings = []
    for idx, h in enumerate(headings):
        if h["type"] in ("TOPIC", "SUBTOPIC"):
            is_toc = False
            # Look backwards
            for prev_h in reversed(headings[:idx]):
                if prev_h["page"] != h["page"]:
                    break
                if prev_h["type"] in ("TOPIC", "SUBTOPIC"):
                    if h["line_idx"] - prev_h["line_idx"] < 5:
                        is_toc = True
                    break
            # Look forwards
            for next_h in headings[idx + 1:]:
                if next_h["page"] != h["page"]:
                    break
                if next_h["type"] in ("TOPIC", "SUBTOPIC"):
                    if next_h["line_idx"] - h["line_idx"] < 5:
                        is_toc = True
                    break
            if is_toc:
                continue
        filtered_headings.append(h)

    return filtered_headings


"""PDF extraction — line-level with font metadata and sanitation."""

from dataclasses import dataclass

import pymupdf




@dataclass
class DocLine:
    page: int
    line_no: int
    global_idx: int
    text: str
    font_size: float = 0.0
    is_bold: bool = False
    char_start: int = 0
    char_end: int = 0


def build_line_char_offsets(doc_lines: list[DocLine]) -> list[int]:
    """Character offset at the start of each line in joined document text."""
    offsets: list[int] = []
    pos = 0
    for i, ln in enumerate(doc_lines):
        ln.char_start = pos
        offsets.append(pos)
        pos += len(ln.text)
        ln.char_end = pos
        if i < len(doc_lines) - 1:
            pos += 1
    return offsets


def line_range_char_span(doc_lines: list[DocLine], start: int, end: int) -> tuple[int, int]:
    """Map exclusive line range [start, end) to [char_start, char_end) in document text."""
    if not doc_lines or start >= end or start >= len(doc_lines):
        return (0, 0)
    end = min(end, len(doc_lines))
    cs = doc_lines[start].char_start
    last = doc_lines[end - 1]
    ce = last.char_end
    return (cs, ce)


def extract_document(pdf_path: str) -> tuple[str, list[DocLine], list[tuple[int, str]]]:
    doc = pymupdf.open(pdf_path)
    lines: list[DocLine] = []
    pages: list[tuple[int, str]] = []
    parts: list[str] = []
    gidx = 0

    for page_idx, page in enumerate(doc):
        pnum = page_idx + 1
        page_lines: list[str] = []

        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                raw = "".join(s.get("text", "") for s in spans)
                text = sanitize_line(raw)
                if is_garbage_line(text):
                    continue
                fs = max((s.get("size", 0) for s in spans), default=0)
                bold = any("bold" in (s.get("font", "") or "").lower() for s in spans)
                page_lines.append(text)
                lines.append(DocLine(pnum, len(page_lines), gidx, text, fs, bold))
                parts.append(text)
                gidx += 1

        pages.append((pnum, "\n".join(page_lines)))

    doc.close()
    build_line_char_offsets(lines)
    return "\n".join(parts), lines, pages


def lines_to_text(doc_lines: list[DocLine], start: int, end: int) -> str:
    return "\n".join(doc_lines[i].text for i in range(start, min(end, len(doc_lines)))).strip()


"""Semantic chunking — one concept per CONTENT node, structural titles only."""

import re






TOPIC_LINE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$")
SUBTOPIC_LINE = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")
FORM_LINE = re.compile(
    r"^(?:\d+\.\d+\s+)?(?:FORM\s+)?(?:GST\s+)?(?:REG|GSTR|CMP|PMT|DRC)",
    re.I,
)
CONCEPT_MARKERS = [
    RE_BOOK_SECTION,
    re.compile(r"^\d+\.\d+(?:\.\d+)?\s+(?:FORM\s+)?(?:GST|GSTR)", re.I),
    re.compile(r"^\d+\.\d+\s+GSTR", re.I),
    re.compile(r"^\d+\.\d+\s+(?:Let\s+us\s+sum\s+up|Let\s+Sum\s+Up|Test\s+Your)", re.I),
    re.compile(r"^FORM\s+GST\s+(?:CMP|PMT|REG)", re.I),
]

MULTI_TOPIC_PATTERNS = [
    ("gstr1", re.compile(r"gstr[- ]?1\b", re.I)),
    ("gstr2a", re.compile(r"gstr[- ]?2a", re.I)),
    ("gstr2b", re.compile(r"gstr[- ]?2b", re.I)),
    ("gstr3b", re.compile(r"gstr[- ]?3b", re.I)),
    ("reg06", re.compile(r"reg[- ]?0?6", re.I)),
    ("cmp08", re.compile(r"cmp[- ]?0?8", re.I)),
    ("pmt06", re.compile(r"pmt[- ]?0?6", re.I)),
    ("drc03", re.compile(r"drc[- ]?0?3", re.I)),
]


def resolve_title(doc_lines: list[DocLine], start: int, end: int, fallback: str) -> str:
    for i in range(start, min(end, start + 8)):
        line = doc_lines[i].text.strip()
        if is_structural_heading_line(line):
            return clean_title(line, "CONTENT")
    for i in range(start, end):
        line = doc_lines[i].text.strip()
        if is_structural_heading_line(line):
            return clean_title(line, "CONTENT")
    if fallback and not is_paragraph_title(fallback):
        return clean_title(fallback, "CONTENT")
    return clean_title(fallback, "CONTENT")[:80] if fallback else "Content"


def _boundary_indices(doc_lines: list[DocLine], start: int, end: int) -> list[int]:
    indices = [start]
    for i in range(start + 1, end):
        line = doc_lines[i].text.strip()
        if not line:
            continue
        if SUBTOPIC_LINE.match(line) and len(line) < 100:
            indices.append(i)
        elif re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", line) and len(line) < 100:
            indices.append(i)
        elif TOPIC_LINE.match(line) and len(line) < 100 and is_structural_heading_line(line):
            indices.append(i)
        elif FORM_LINE.match(line) and len(line) < 90:
            indices.append(i)
        elif any(p.match(line) for p in CONCEPT_MARKERS):
            indices.append(i)
    indices.append(end)
    # dedupe sorted
    out = [indices[0]]
    for x in indices[1:]:
        if x > out[-1]:
            out.append(x)
    return out


def _count_concepts(text: str) -> int:
    """Count distinct form/topic markers at line starts (not cross-references in prose)."""
    found = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 120:
            continue
        if not (TOPIC_LINE.match(line) or FORM_LINE.match(line) or SUBTOPIC_LINE.match(line)):
            continue
        for name, pat in MULTI_TOPIC_PATTERNS:
            if pat.search(line):
                found.add(name)
    return len(found)


def _split_oversized(
    doc_lines: list[DocLine],
    title: str,
    start: int,
    end: int,
    max_chars: int,
) -> list[tuple[str, str, int, int]]:
    raw = sanitize_raw_content(lines_to_text(doc_lines, start, end))
    if len(raw) <= max_chars and _count_concepts(raw) <= 1:
        return [(title, raw, start, end)] if len(raw) >= 20 else []

    boundaries = _boundary_indices(doc_lines, start, end)
    if len(boundaries) <= 2:
        # force split by char windows at paragraph breaks
        mid = start + (end - start) // 2
        if mid > start and mid < end:
            boundaries = [start, mid, end]
        else:
            return [(title, raw[:max_chars], start, end)] if raw else []

    chunks: list[tuple[str, str, int, int]] = []
    for j in range(len(boundaries) - 1):
        s, e = boundaries[j], boundaries[j + 1]
        if e <= s:
            continue
        t = resolve_title(doc_lines, s, e, title)
        part_raw = sanitize_raw_content(lines_to_text(doc_lines, s, e))
        if len(part_raw) < 20:
            continue
        if len(part_raw) > max_chars or _count_concepts(part_raw) > 1:
            sub = _split_oversized(doc_lines, t, s, e, max_chars)
            chunks.extend(sub)
        else:
            chunks.append((t, part_raw, s, e))
    return chunks


def _filter_adjacent_overlap(
    chunks: list[tuple[str, str, int, int]],
    threshold: float = 0.15,
) -> list[tuple[str, str, int, int]]:
    if len(chunks) <= 1:
        return chunks
    out: list[tuple[str, str, int, int]] = [chunks[0]]
    for item in chunks[1:]:
        prev_title, prev_raw, prev_s, prev_e = out[-1]
        curr_title, curr_raw, curr_s, curr_e = item
        
        if (
            prev_title.strip().lower() != curr_title.strip().lower()
            and (is_structural_heading_line(curr_title) or is_structural_heading_line(prev_title))
        ):
            out.append(item)
            continue
        if jaccard_similarity(prev_raw, curr_raw) >= threshold:
            merged_raw = prev_raw + "\n" + curr_raw
            merged_e = max(prev_e, curr_e)
            out[-1] = (prev_title, merged_raw, prev_s, merged_e)
        else:
            out.append(item)
    return out


def semantic_chunks(
    doc_lines: list[DocLine],
    start: int,
    end: int,
    default_title: str = "Content",
    max_chars: int = 6000,
    overlap_adjacent_threshold: float = 0.15,
) -> list[tuple[str, str, int, int]]:
    if start >= end or start >= len(doc_lines):
        return []

    boundaries = _boundary_indices(doc_lines, start, end)
    if len(boundaries) <= 2:
        title = resolve_title(doc_lines, start, end, default_title)
        if is_synthetic_title(title) and not re.match(r"^\d+\.\d+", title):
            return []
        chunks = _split_oversized(doc_lines, title, start, end, max_chars)
        return _filter_adjacent_overlap(chunks, overlap_adjacent_threshold)

    out: list[tuple[str, str, int, int]] = []
    for j in range(len(boundaries) - 1):
        s, e = boundaries[j], boundaries[j + 1]
        if e <= s:
            continue
        title = resolve_title(doc_lines, s, e, default_title)
        if is_synthetic_title(title) and not re.match(r"^\d+\.\d+", title):
            continue
        out.extend(_split_oversized(doc_lines, title, s, e, max_chars))
    return _filter_adjacent_overlap(out, overlap_adjacent_threshold)


"""Content deduplication: SHA256 exact hash + overlap rejection."""

from dataclasses import dataclass




@dataclass
class DedupStats:
    rejected_exact: int = 0
    rejected_overlap: int = 0
    rejected_parent_child: int = 0
    accepted: int = 0


class ContentDeduplicator:
    def __init__(self, overlap_threshold: float = 0.85):
        self.overlap_threshold = overlap_threshold
        self._hashes: dict[str, str] = {}
        self._norm_samples: list[str] = []
        self.stats = DedupStats()

    def register(self, raw_content: str) -> tuple[bool, str]:
        norm = normalize_for_hash(raw_content)
        if len(norm) < 20:
            return False, ""

        h = sha256_content(raw_content)
        if h in self._hashes:
            self.stats.rejected_exact += 1
            return False, h

        for prev in self._norm_samples:
            if jaccard_similarity(norm, prev) >= self.overlap_threshold:
                self.stats.rejected_overlap += 1
                return False, h

        self._hashes[h] = norm[:80]
        self._norm_samples.append(norm[:5000])
        self.stats.accepted += 1
        return True, h

    def child_duplicates_parent(self, parent_raw: str, child_raw: str) -> bool:
        """Parent must not embed full child text."""
        if not parent_raw or not child_raw:
            return False
        pn = normalize_for_hash(parent_raw)
        cn = normalize_for_hash(child_raw)
        if len(cn) < 40:
            return False
        if cn in pn:
            return True
        return jaccard_similarity(pn, cn) >= 0.92


"""Rule-based semantic compression (60–80% retention, not summarization)."""

import re

DROP_PATTERNS = [
    re.compile(r"^the study material has been prepared", re.I),
    re.compile(r"^all copyrights with", re.I),
    re.compile(r"^self-instructional study material", re.I),
    re.compile(r"^jagat guru nanak dev", re.I),
    re.compile(r"^established by act no", re.I),
    re.compile(r"^in keeping with the nature", re.I),
    re.compile(r"^we,? at the university,? welcome", re.I),
    re.compile(r"^prof\.\s", re.I),
    re.compile(r"^let us sum up", re.I),
    re.compile(r"^test your knowledge", re.I),
]

KEEP_BOOST = re.compile(
    r"\b(GST|GSTR|REG-?\d|form|section|rule|shall|must|due date|portal|"
    r"registration|certificate|challan|return|filing|tax|invoice|credit|"
    r"\d+%|\d{4}|Rs\.|rupee|deadline|procedure|step|download|upload|DRC|CMP|PMT)\b",
    re.I,
)


def compression_ratio(raw: str, compressed: str) -> float:
    if not raw:
        return 0.0
    return len(compressed or "") / len(raw)


def _split_sentences(text: str) -> list[str]:
    # Replace newlines with spaces to join split layout lines
    normalized_text = re.sub(r"\s+", " ", text or "")
    parts = re.split(r"(?<=[.!?])\s+", normalized_text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]


def _sentence_score(s: str) -> float:
    score = 1.0
    if KEEP_BOOST.search(s):
        score += 3.0
    if re.search(r"\d", s):
        score += 1.5
    if len(s) < 30:
        score -= 0.5
    if any(p.search(s) for p in DROP_PATTERNS):
        score -= 10.0
    return score


def compress_text(raw: str, target_ratio: float = 0.70, min_ratio: float = 0.60) -> str:
    if not raw or not raw.strip():
        return ""

    sentences = _split_sentences(raw)
    if not sentences:
        return raw.strip()

    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.lower())[:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    if not unique:
        return raw.strip()

    target_len = int(len(raw) * target_ratio)
    min_len = int(len(raw) * min_ratio)

    scored = sorted(unique, key=_sentence_score, reverse=True)
    kept: list[str] = []
    total = 0

    for s in scored:
        if _sentence_score(s) < 0:
            continue
        if total + len(s) > target_len and total >= min_len:
            break
        kept.append(s)
        total += len(s) + 1

    if total < min_len:
        for s in scored:
            if s not in kept:
                if total + len(s) > int(len(raw) * 0.85) and kept:
                    break
                kept.append(s)
                total += len(s) + 1
                if total >= min_len:
                    break

    order = {id(s): i for i, s in enumerate(unique)}
    kept.sort(key=lambda s: order.get(id(s), 999))

    result = "\n".join(kept).strip() if kept else raw.strip()
    if len(raw) > 400 and len(result) / len(raw) > 0.92:
        capped: list[str] = []
        total = 0
        for s in unique:
            if total + len(s) > target_len and total >= min_len:
                break
            capped.append(s)
            total += len(s) + 1
        if capped:
            result = "\n".join(capped).strip()
        if len(result) / len(raw) > 0.92:
            budget = max(int(len(raw) * target_ratio), min_len)
            words = result.split()
            trimmed = []
            wlen = 0
            for w in words:
                if wlen + len(w) + 2 > budget and trimmed:
                    break
                trimmed.append(w)
                wlen += len(w) + 1
            if trimmed:
                result = " ".join(trimmed).strip()
                if result and result[-1] not in ('.', '?', '!'):
                    result = result.rstrip(' .!?') + "."
    return result


"""Micro-summary generation (1–3 sentences, metadata only)."""


def micro_summary_from_content(title: str, raw: str, max_sentences: int = 3) -> str:
    """Deterministic metadata summary (1–3 sentences) from raw_content only."""
    raw = (raw or "").strip()
    if not raw:
        return f"Section: {title}."

    picked: list[str] = []
    t = title.strip()
    if re.match(r"^\d+\.\d+", t):
        picked.append(f"Covers {t}.")

    sents = _split_sentences(raw)
    for s in sents:
        if len(s) < 20:
            continue
        if any(
            k in s.lower()
            for k in ("gst", "gstr", "form", "registration", "return", "portal", "tax", "filing", "rule", "act")
        ):
            picked.append(s if len(s) <= 220 else s[:217] + "...")
        if len(picked) >= max_sentences:
            break

    if len(picked) < max_sentences:
        for s in sents:
            if s not in picked:
                picked.append(s if len(s) <= 220 else s[:217] + "...")
            if len(picked) >= max_sentences:
                break

    if not picked:
        picked.append(f"Section: {title}.")

    return " ".join(picked[:max_sentences]).strip()


def container_micro_summary(title: str, child_titles: list[str]) -> str:
    kids = ", ".join(child_titles[:6])
    if kids:
        return f"{title}: includes {kids}."
    return f"{title} (structural section)."
