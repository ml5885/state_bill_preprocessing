"""
Remove boilerplate text from state legislative bills.

Each state has common boilerplate sentences that appear across many bills.
This module identifies and removes them by computing document frequency of
normalized sentences within each state's corpus.
"""
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Regex for splitting text into candidate sentences
SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}|;\s+')
LINE_NUM_RE = re.compile(r"(?m)(^[ \t]*\d{1,4}[.)]?[ \t]+)|([ \t]+\d{1,4}[.)]?[ \t]*$)")
BOUNDARY_WS_AFTER_PUNCT_RE = re.compile(r'(?<=[.!?;])\s+')
BOUNDARY_NEWLINE_RUN_RE = re.compile(r'\n{2,}')

# Regex patterns for detecting XML/HTML/PDF metadata junk
XML_TAG_RE = re.compile(r'<[^>]+>')
PDF_METADATA_RE = re.compile(r'^\s*/[A-Za-z]+\s+', re.MULTILINE)
DICT_MARKER_RE = re.compile(r'<<|>>')


def sentences(text, min_chars=40, min_tokens=6, max_chars=5000):
    """Split text into candidate sentences for boilerplate detection."""
    parts = [p.strip() for p in SPLIT_RE.split(text.replace("\r\n", "\n").replace("\r", "\n")) if p.strip()]
    out = []
    for s in parts:
        if len(s) < min_chars or len(s) > max_chars or len(s.split()) < min_tokens:
            continue
        out.append(s)
    return out


def normalize_sentence(sentence):
    """Normalize a sentence for boilerplate detection."""
    sentence = sentence.lower()
    sentence = re.sub(r'\d+', ' num ', sentence)
    sentence = re.sub(r'[^a-z\s]', ' ', sentence)
    return " ".join(sentence.split())


def is_metadata_junk(text):
    """
    Detect if text is likely XML/HTML/PDF metadata junk.
    
    Returns True if the text appears to be technical metadata that should be removed.
    """
    if not text or not text.strip():
        return False
    
    text_stripped = text.strip()
    lines = text_stripped.split('\n')
    
    # Check for high density of XML tags
    tag_count = len(XML_TAG_RE.findall(text_stripped))
    if tag_count > 5:  # More than 5 XML tags
        return True
    
    # Check for PDF metadata patterns
    pdf_lines = 0
    dict_markers = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Lines starting with /PropertyName
        if PDF_METADATA_RE.match(line):
            pdf_lines += 1
        
        # Check for dictionary markers
        if DICT_MARKER_RE.search(line):
            dict_markers += 1
        
        # Check for patterns like /Property value or /Property
        if line_stripped.startswith('/') and len(line_stripped.split()) <= 3:
            pdf_lines += 1
    
    # If more than 30% of non-empty lines look like PDF metadata, it's junk
    non_empty_lines = len([l for l in lines if l.strip()])
    if non_empty_lines > 0:
        pdf_ratio = pdf_lines / non_empty_lines
        if pdf_ratio > 0.3:
            return True
    
    # If we have multiple dictionary markers, likely PDF metadata
    if dict_markers >= 3:
        return True
    
    # Check for common PDF/XML keywords in high density
    junk_keywords = ['/sRGBProfile', '/ColorConversion', '/EmbedFont', 
                     '/CompressObjects', '/ImageMemory', '/ParseDSC',
                     '/QFactor', '/HSamples', '/VSamples', '/TileWidth',
                     'Arial-Black', 'TimesNewRoman', 'CourierNew']
    
    keyword_count = sum(1 for kw in junk_keywords if kw in text_stripped)
    if keyword_count >= 3:
        return True
    
    return False


def clean_metadata_junk(text):
    """
    Remove XML/HTML tags and PDF metadata junk from text.
    
    This is applied before boilerplate detection to remove technical artifacts.
    """
    if not text:
        return text
    
    # Remove XML/HTML tags
    text = XML_TAG_RE.sub(' ', text)
    
    # Split into lines and filter out PDF metadata
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip lines that start with PDF properties
        if PDF_METADATA_RE.match(line):
            continue
        
        # Skip lines with dictionary markers if they're short
        if DICT_MARKER_RE.search(line) and len(line_stripped.split()) <= 5:
            continue
        
        # Skip very short lines that start with /
        if line_stripped.startswith('/') and len(line_stripped.split()) <= 2:
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_hardcoded_boilerplate(text):
    """
    Remove specific hardcoded boilerplate text patterns.
    
    This includes:
    - Membership/compensation tables
    - Census/redistricting number lists
    - Tabs not preceded by newlines (preserve \n\t+ patterns for chunking)
    - Long runs of periods
    """
    if not text:
        return text
    
    # Remove tab characters NOT preceded by newline (preserve \n\t+ patterns)
    # These \n\t+ patterns are important paragraph markers for chunking
    text = re.sub(r'(?<!\n)\t+', ' ', text)
    
    # Remove runs of 4 or more periods
    text = re.sub(r'\.{4,}', ' ', text)
    
    # Remove specific hardcoded text patterns (case-insensitive)
    # Membership table boilerplate
    membership_pattern = re.compile(
        r'Membership\s+Class\s*Percentage\s+of\s*Gross\s*Compensation\s*,?\s*Effective\s+July\s+1,\s*2011\s*2009',
        re.IGNORECASE
    )
    text = membership_pattern.sub('', text)
    
    # Remove census/redistricting number lists
    # Florida pattern: "to be designated by such numbers as follows:"
    florida_redistricting = re.compile(
        r'to\s+be\s+designated\s+by\s+such\s+numbers\s+as\s+follows:.*?(?=\n\s*\([a-z]\)|\n\s*Section|\n\s*SECTION|$)',
        re.IGNORECASE | re.DOTALL
    )
    text = florida_redistricting.sub('', text)
    
    # Michigan pattern: "For terms of office beginning on or after"
    michigan_redistricting = re.compile(
        r'For\s+terms\s+of\s+office\s+beginning\s+on\s+or\s+after\s+[^:]+:\s*the\s+districts\s+are\s+constituted\s+and\s+numbered\s+as\s+follows:.*?(?=\n\s*\([a-z]\)|\n\s*Section|\n\s*SECTION|$)',
        re.IGNORECASE | re.DOTALL
    )
    text = michigan_redistricting.sub('', text)
    
    # Remove long runs of enumerated items that are likely redistricting data
    # Pattern: blocks/tracts with many numbers in sequence
    # Look for patterns like "blocks 1234, 5678, 9012" with many numbers
    redistricting_blocks = re.compile(
        r'(?:blocks?|tracts?)\s+\d+(?:\s*,\s*\d+){20,}',  # 20+ comma-separated numbers
        re.IGNORECASE
    )
    text = redistricting_blocks.sub('', text)
    
    # Remove voting tabulation district lists
    vtd_pattern = re.compile(
        r'(?:All\s+of\s+)?voting\s+tabulation\s+districts?\s+\d+(?:\s*,\s*\d+){10,}',
        re.IGNORECASE
    )
    text = vtd_pattern.sub('', text)
    
    # Clean up multiple spaces and newlines left by removals
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def remove_boilerplate(text, boilerplate_keys):
    """Remove boilerplate sentences and line numbers from bill text."""
    if not text:
        return (0, 0, "")

    # First, clean XML/HTML/PDF metadata junk
    text = clean_metadata_junk(text)
    
    # Remove hardcoded boilerplate patterns
    text = remove_hardcoded_boilerplate(text)

    text_no_linenum = LINE_NUM_RE.sub('', text)
    sents = sentences(text_no_linenum)
    if not sents:
        return (0, 0, "")

    words_before = sum(len(s.split()) for s in sents)

    # Split by sentence and paragraph boundaries, preserving whitespace
    boundaries = []
    for m in BOUNDARY_WS_AFTER_PUNCT_RE.finditer(text_no_linenum):
        boundaries.append(m.start())
    for m in BOUNDARY_NEWLINE_RUN_RE.finditer(text_no_linenum):
        boundaries.append(m.start())
    boundaries = sorted(set(boundaries))

    spans = []
    prev = 0
    for b in boundaries:
        spans.append((prev, b))
        prev = b
    spans.append((prev, len(text_no_linenum)))

    kept_parts = []
    for start, end in spans:
        segment = text_no_linenum[start:end]
        if not segment.strip():
            kept_parts.append(segment)
            continue
        normalized = normalize_sentence(segment)
        if normalized in boilerplate_keys:
            stripped = segment.strip()
            if len(stripped) >= 40 and len(stripped.split()) >= 6:
                continue
        kept_parts.append(segment)

    cleaned_text = "".join(kept_parts)
    kept_sents = sentences(cleaned_text)
    words_after = sum(len(s.split()) for s in kept_sents)

    return (words_before, words_after, cleaned_text)


@dataclass
class CleanedDoc:
    state: str
    session: str
    bill_id: str
    unique_id: str
    sunlight_id: str
    bill_version: str
    bill_text: str
    words_before: int
    words_after: int


def process_bills(input_path, min_ratio=0.10, min_docs=5, states=None):
    """
    Process raw state bills and remove boilerplate per state.
    
    Args:
        input_path: Path to NDJSON (.gz) file with state bills
        min_ratio: Minimum fraction of docs that must contain a sentence for it to be boilerplate
        min_docs: Minimum absolute number of docs that must contain a sentence
        states: Optional list of state codes to process (case-insensitive)
    
    Returns:
        (DataFrame, stats_dict) where stats_dict has per-state word counts
    """
    inp = Path(input_path)
    if not inp.exists():
        alt = Path(str(input_path) + '.gz')
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

    # Exclude Colorado, Puerto Rico, and DC from all processing
    excluded_states = {'co', 'pr', 'dc'}
    allowed_states = set(s.strip().lower() for s in states) if states else None

    current_state = None
    sample_texts = []
    state_records = []
    per_state_stats = {}
    cleaned_rows = []

    def finalize_state(state):
        nonlocal sample_texts, state_records, cleaned_rows

        if not state:
            return

        # Skip excluded states
        if state in excluded_states:
            sample_texts = []
            state_records = []
            return

        if allowed_states is not None and state not in allowed_states:
            sample_texts = []
            state_records = []
            return

        # Build document frequency over normalized sentences
        docs = [sentences(t) for t in sample_texts if sentences(t)]
        df = {}
        rep = {}
        for sents in docs:
            keys = set()
            for s in sents:
                k = normalize_sentence(s)
                if not k:
                    continue
                if k not in rep or len(s) > len(rep[k]):
                    rep[k] = s
                keys.add(k)
            for k in keys:
                df[k] = df.get(k, 0) + 1

        n = len(docs)
        boilerplate_keys = set()
        for k, d in df.items():
            ratio = d / n if n else 0.0
            if d >= min_docs and ratio >= min_ratio:
                boilerplate_keys.add(k)

        # Clean all documents for this state
        docs_count = 0
        bills_with_text = 0
        bills_total = len(state_records)
        words_before = 0
        words_after = 0

        for obj in state_records:
            # Only process the last version of each bill
            b, a, c = remove_boilerplate(obj.get("bill_document_last"), boilerplate_keys)
            if b > 0:
                docs_count += 1
                words_before += b
                words_after += a
                bills_with_text += 1
                if c:
                    cleaned_rows.append(CleanedDoc(
                        state=state,
                        session=obj.get("session"),
                        bill_id=obj.get("bill_id"),
                        unique_id=obj.get("unique_id"),
                        sunlight_id=obj.get("sunlight_id"),
                        bill_version="last",
                        bill_text=c,
                        words_before=b,
                        words_after=a,
                    ))

        per_state_stats[state] = {
            "bills_total": bills_total,
            "bills_with_text": bills_with_text,
            "documents": docs_count,
            "words_before": words_before,
            "words_after": words_after,
        }

        sample_texts = []
        state_records = []

    opener = gzip.open if inp.suffix.lower() == ".gz" else open
    mode = "rt" if inp.suffix.lower() == ".gz" else "r"

    with opener(inp, mode, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Processing bills", unit="bill"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            state = str(obj.get("state", "")).lower()
            if not state:
                continue

            # Skip excluded states
            if state in excluded_states:
                continue

            if current_state is None:
                current_state = state
            elif state != current_state:
                finalize_state(current_state)
                current_state = state

            if allowed_states is None or state in allowed_states:
                # Only use bill_document_last for consistency
                text = obj.get("bill_document_last")
                if text and text.strip():
                    # Clean metadata junk before adding to sample
                    cleaned = clean_metadata_junk(text)
                    if cleaned and cleaned.strip():
                        sample_texts.append(cleaned)
                state_records.append(obj)

        finalize_state(current_state)

    df = pd.DataFrame([row.__dict__ for row in cleaned_rows])
    return df, per_state_stats


def print_summary_table(stats):
    """Print a summary table of boilerplate removal statistics."""
    header = [
        "state",
        "bills_total",
        "bills_with_text",
        "documents",
        "words_before",
        "words_after",
        "mean_before_per_doc",
        "mean_after_per_doc",
        "pct_reduction_words",
    ]

    total_bills = 0
    total_bills_with_text = 0
    total_docs = 0
    total_before = 0
    total_after = 0
    rows = []
    
    for state in tqdm(sorted(stats.keys()), desc="Preparing summary table"):
        s = stats[state]
        bills_total = s["bills_total"]
        bills_with_text = s["bills_with_text"]
        docs = s["documents"]
        wb = s["words_before"]
        wa = s["words_after"]
        mb = (wb / docs) if docs else 0.0
        ma = (wa / docs) if docs else 0.0
        pr = (100.0 * (wb - wa) / wb) if wb else 0.0

        total_bills += bills_total
        total_bills_with_text += bills_with_text
        total_docs += docs
        total_before += wb
        total_after += wa

        rows.append({
            "state": state,
            "bills_total": bills_total,
            "bills_with_text": bills_with_text,
            "documents": docs,
            "words_before": wb,
            "words_after": wa,
            "mean_before_per_doc": mb,
            "mean_after_per_doc": ma,
            "pct_reduction_words": pr,
        })

    total_pr = (100.0 * (total_before - total_after) / total_before) if total_before else 0.0
    mb = (total_before / total_docs) if total_docs else 0.0
    ma = (total_after / total_docs) if total_docs else 0.0

    rows.append({
        "state": "TOTAL",
        "bills_total": total_bills,
        "bills_with_text": total_bills_with_text,
        "documents": total_docs,
        "words_before": total_before,
        "words_after": total_after,
        "mean_before_per_doc": mb,
        "mean_after_per_doc": ma,
        "pct_reduction_words": total_pr,
    })

    df = pd.DataFrame(rows)[header]
    df["mean_before_per_doc"] = df["mean_before_per_doc"].round(2)
    df["mean_after_per_doc"] = df["mean_after_per_doc"].round(2)
    df["pct_reduction_words"] = df["pct_reduction_words"].round(2)
    print(df.to_string(index=False))
