"""
Remove boilerplate text from state legislative bills.

Each state has common boilerplate sentences that appear across many bills.
This module identifies and removes them by computing document frequency of
normalized sentences within each state's corpus.
"""
import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import chunking

# Regex for splitting text into candidate sentences
SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}|;\s+')
LINE_NUM_RE = re.compile(r"(?m)(^[ \t]*\d{1,4}[.)]?[ \t]+)|([ \t]+\d{1,4}[.)]?[ \t]*$)")
BOUNDARY_WS_AFTER_PUNCT_RE = re.compile(r'(?<=[.!?;])\s+')
BOUNDARY_NEWLINE_RUN_RE = re.compile(r'\n{2,}')

# Regex patterns for detecting XML/HTML/PDF metadata junk
XML_TAG_RE = re.compile(r'<[^>]+>')
PDF_METADATA_RE = re.compile(r'^\s*/[A-Za-z]+\s+', re.MULTILINE)
DICT_MARKER_RE = re.compile(r'<<|>>')

# Common boilerplate section titles to remove
# From Table 13 of [Learning Bill Similarity with Annotated and Augmented Corpora of Bills](https://aclanthology.org/2021.emnlp-main.787/) (Kim et al., EMNLP 2021)
BOILERPLATE_SECTION_TITLES = [
    r"Effective\s+Date",
    r"Authorization\s+of\s+Appropriations",
    r"Vacancies",
    r"Termination",
    r"Table\s+of\s+Contents",
    r"Short\s+Title",
    r"Reference",
    r"Sunset",
    r"Appropriation",
    r"Severability",
    r"Matching\s+Requirement",
    r"Definitions",
]

# Matches: Start of string, optional "Section 1." or "Sec 1." prefix, then the title, then end of line or punctuation
BOILERPLATE_TITLE_RE = re.compile(
    r'^\s*(?:(?:Section|Sec\.|SECTION|SEC\.)\s*\d+[a-zA-Z0-9\.]*\s*)?' # Optional section number
    r'(?:' + '|'.join(BOILERPLATE_SECTION_TITLES) + r')' # The Title
    r'(?:[:\.]|\s+[-–—]+\s+|\s*$)', # Punctuation or end of line
    re.IGNORECASE | re.MULTILINE
)


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


def clean_text_pre_normalization(text):
    """Convert non-breaking spaces (\xa0), en-spaces, etc., into standard spaces."""
    if not text:
        return ""
    # Normalize unicode characters (e.g., turn \xa0 into space)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\xa0', ' ')
    return text


def is_metadata_junk(text):
    """Detect if text is likely XML/HTML/PDF metadata junk."""
    if not text or not text.strip():
        return False
    
    text_stripped = text.strip()
    lines = text_stripped.split('\n')
    
    tag_count = len(XML_TAG_RE.findall(text_stripped))
    if tag_count > 5:
        return True
    
    pdf_lines = 0
    dict_markers = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if PDF_METADATA_RE.match(line):
            pdf_lines += 1
        if DICT_MARKER_RE.search(line):
            dict_markers += 1
        if line_stripped.startswith('/') and len(line_stripped.split()) <= 3:
            pdf_lines += 1
    
    non_empty_lines = len([l for l in lines if l.strip()])
    if non_empty_lines > 0:
        if (pdf_lines / non_empty_lines) > 0.3:
            return True
    
    if dict_markers >= 3:
        return True
    
    junk_keywords = ['/sRGBProfile', '/ColorConversion', '/EmbedFont', 
                     '/CompressObjects', '/ImageMemory', '/ParseDSC',
                     'Arial-Black', 'TimesNewRoman']
    
    keyword_count = sum(1 for kw in junk_keywords if kw in text_stripped)
    if keyword_count >= 3:
        return True
    
    return False


def clean_metadata_junk(text):
    """Remove XML/HTML tags and PDF metadata junk from text."""
    if not text:
        return text
    
    text = XML_TAG_RE.sub(' ', text)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if PDF_METADATA_RE.match(line):
            continue
        if DICT_MARKER_RE.search(line) and len(line_stripped.split()) <= 5:
            continue
        if line_stripped.startswith('/') and len(line_stripped.split()) <= 2:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_massachusetts_boilerplate(text):
    """Remove Massachusetts-specific boilerplate patterns."""
    if not text:
        return text
    
    header_pattern = re.compile(
        r'(?:FILED\s+ON:\s*\d{1,2}/\d{1,2}/\d{4})?\s*'
        r'(?:SENATE|HOUSE)\s*'
        r'(?:DOCKET,?\s*(?:NO\.|NUM)?)?'
        r'(?:[\.\s]+)?'
        r'(?:No\.|Num)\s*\w+',
        re.IGNORECASE
    )
    text = header_pattern.sub(' ', text)

    petition_block_pattern = re.compile(
        r'PETITION\s+OF:.*?District/Address:.*?'
        r'(?=\n\s*By\s+(?:Ms\.|Mr\.|Mrs\.|Rep\.|Sen\.)|An\s+Act|Be\s+it\s+enacted|FILED\s+ON:)',
        re.IGNORECASE | re.DOTALL
    )
    text = petition_block_pattern.sub(' ', text)

    by_line_pattern = re.compile(
        r'By\s+(?:Ms\.|Mr\.|Mrs\.|Rep\.|Sen\.)\s+[^,]+,\s+a\s+petition\s+\(accompanied\s+by\s+[^)]+\).*?(?=\n|An\s+Act|Be\s+it\s+enacted)',
        re.IGNORECASE | re.DOTALL
    )
    text = by_line_pattern.sub(' ', text)

    undersigned_pattern = re.compile(
        r'The\s+undersigned\s+(?:legislators|citizens).*?(?=An\s+Act|Be\s+it\s+enacted|PETITION\s+OF)',
        re.IGNORECASE | re.DOTALL
    )
    text = undersigned_pattern.sub(' ', text)

    formal_address_pattern = re.compile(
        r'To\s+the\s+Honorable\s+Senate\s+and\s+House.*?assembled:',
        re.IGNORECASE | re.DOTALL
    )
    text = formal_address_pattern.sub(' ', text)

    presented_by_pattern = re.compile(
        r'PRESENTED\s+BY:\s*[^\n]+',
        re.IGNORECASE
    )
    text = presented_by_pattern.sub(' ', text)

    commonwealth_pattern = re.compile(
        r'^\s*The\s+Commonwealth\s+of\s+Massachusetts\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    text = commonwealth_pattern.sub(' ', text)
    text = re.sub(r'^\s*The\s+Commonwealth\s+of\s+Massachusetts\s+(?=\w)', '', text, flags=re.IGNORECASE | re.MULTILINE)

    return text


def remove_hardcoded_boilerplate(text):
    """Remove hardcoded boilerplate text patterns."""
    if not text:
        return text
    
    text = re.sub(r'[ \t\xa0]+', ' ', text)
    text = re.sub(r'_{3,}', ' ', text)
    text = re.sub(r'\.{4,}', ' ', text)
    
    text = remove_massachusetts_boilerplate(text)

    membership_pattern = re.compile(
        r'Membership\s+Class\s*Percentage\s+of\s*Gross\s*Compensation\s*,?\s*Effective\s+July\s+1,\s*2011\s*2009',
        re.IGNORECASE
    )
    text = membership_pattern.sub('', text)
    
    florida_redistricting = re.compile(
        r'to\s+be\s+designated\s+by\s+such\s+numbers\s+as\s+follows:.*?(?=\n\s*\([a-z]\)|\n\s*Section|\n\s*SECTION|$)',
        re.IGNORECASE | re.DOTALL
    )
    text = florida_redistricting.sub('', text)

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def remove_boilerplate(text, boilerplate_keys):
    """Remove boilerplate sentences and line numbers from bill text."""
    if not text:
        return (0, 0, "")

    text = clean_metadata_junk(text)
    text = clean_text_pre_normalization(text)
    text = remove_hardcoded_boilerplate(text)
    text_no_linenum = LINE_NUM_RE.sub('', text)
    
    words_before = len(text_no_linenum.split())

    sents = sentences(text_no_linenum)
    if not sents:
        return (words_before, words_before, text_no_linenum)

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
            # Require minimum length to avoid deleting short non-boilerplate phrases
            if len(stripped) >= 40 and len(stripped.split()) >= 6:
                continue
        kept_parts.append(segment)

    cleaned_text = "".join(kept_parts)
    words_after = len(cleaned_text.split())

    return (words_before, words_after, cleaned_text)


def remove_boilerplate_sections(text, state=None):
    """Remove boilerplate sections based on titles (e.g., Effective Date, Severability)."""
    if not text:
        return text

    # Use the chunker to split into logical sections/paragraphs
    chunks = chunking.split_into_chunks(text, state=state)
    
    kept_chunks = []
    for chunk in chunks:
        if BOILERPLATE_TITLE_RE.match(chunk):
            continue
        kept_chunks.append(chunk)

    return "\n\n".join(kept_chunks)


def remove_boilerplate_via_chunking(text, boilerplate_keys, min_chars=100, max_chars=100, state=None):
    """
    Remove boilerplate by chunking text into small pieces and filtering them.
    
    Returns:
        (words_before, words_after, cleaned_text)
    """
    if not text:
        return (0, 0, "")

    text = LINE_NUM_RE.sub('', text)

    chunks = chunking.split_into_chunks(text, state=state, min_chars=min_chars, max_chars=max_chars)
    
    if not chunks:
        return (0, 0, text)

    words_before = len(text.split())
    
    kept_chunks = []
    for c in chunks:
        if not c.strip():
            continue
        
        normalized = normalize_sentence(c)
        if normalized in boilerplate_keys:
            continue
            
        kept_chunks.append(c)

    cleaned_text = "\n\n".join(kept_chunks)
    words_after = len(cleaned_text.split())
    
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


def process_bills(input_path, min_ratio=0.10, min_docs=5, states=None, 
                 tiny_chunk_min=100, tiny_chunk_max=100,
                 aggressive_ratio=0.02):
    """
    Process raw state bills and remove boilerplate per state using multiple methods:
    1. Sentence-based frequency (Standard)
    2. Section-title based removal
    3. Tiny-chunk based frequency (Aggressive)
    """
    inp = Path(input_path)
    if not inp.exists():
        alt = Path(str(input_path) + '.gz')
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

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

        if state in excluded_states:
            sample_texts = []
            state_records = []
            return

        if allowed_states is not None and state not in allowed_states:
            sample_texts = []
            state_records = []
            return

    # --- Phase 1: Build Frequencies ---
        pre_cleaned_samples = []
        for t in sample_texts:
            t = clean_metadata_junk(t)
            t = clean_text_pre_normalization(t)
            t = remove_hardcoded_boilerplate(t)
            t = LINE_NUM_RE.sub('', t)
            pre_cleaned_samples.append(t)

        docs_sentences = [sentences(t) for t in pre_cleaned_samples if sentences(t)]
        sent_df = {}
        for sents in docs_sentences:
            unique_keys = set()
            for s in sents:
                k = normalize_sentence(s)
                if not k: continue
                unique_keys.add(k)
            for k in unique_keys:
                sent_df[k] = sent_df.get(k, 0) + 1

        n_sents = len(docs_sentences)
        boilerplate_sent_keys = set()
        for k, d in sent_df.items():
            ratio = d / n_sents if n_sents else 0.0
            if d >= min_docs and ratio >= min_ratio:
                boilerplate_sent_keys.add(k)

        # 1b. Tiny Chunk Frequency
        chunk_df = {}
        total_docs_chunked = 0
        
        for t in pre_cleaned_samples:
            chunks = chunking.split_into_chunks(t, state=state, min_chars=tiny_chunk_min, max_chars=tiny_chunk_max)
            if not chunks:
                continue
            
            total_docs_chunked += 1
            unique_chunks = set()
            for c in chunks:
                normalized = normalize_sentence(c)
                if normalized:
                    unique_chunks.add(normalized)
            
            for k in unique_chunks:
                chunk_df[k] = chunk_df.get(k, 0) + 1

        boilerplate_chunk_keys = set()
        for k, count in chunk_df.items():
            ratio = count / total_docs_chunked if total_docs_chunked else 0.0
            if count >= min_docs and ratio >= aggressive_ratio:
                boilerplate_chunk_keys.add(k)
        
        print(f"State {state}: Found {len(boilerplate_sent_keys)} sentence keys, {len(boilerplate_chunk_keys)} chunk keys")

        # --- Phase 2: Processing ---
        docs_count = 0
        bills_with_text = 0
        bills_total = len(state_records)
        words_before = 0
        words_after = 0

        for obj in state_records:
            raw_text = obj.get("bill_document_last")
            if not raw_text:
                continue

            wb, wa_step1, text_step1 = remove_boilerplate(raw_text, boilerplate_sent_keys)
            text_step2 = remove_boilerplate_sections(text_step1, state=state)
            
            _, wa_final, text_final = remove_boilerplate_via_chunking(
                text_step2, 
                boilerplate_chunk_keys, 
                min_chars=tiny_chunk_min, 
                max_chars=tiny_chunk_max,
                state=state
            )
            
            if wb > 0:
                docs_count += 1
                words_before += wb
                words_after += wa_final
                bills_with_text += 1
                if text_final:
                    cleaned_rows.append(CleanedDoc(
                        state=state,
                        session=obj.get("session"),
                        bill_id=obj.get("bill_id"),
                        unique_id=obj.get("unique_id"),
                        sunlight_id=obj.get("sunlight_id"),
                        bill_version="last",
                        bill_text=text_final,
                        words_before=wb,
                        words_after=wa_final,
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

            if state in excluded_states:
                continue

            if current_state is None:
                current_state = state
            elif state != current_state:
                finalize_state(current_state)
                current_state = state

            if allowed_states is None or state in allowed_states:
                text = obj.get("bill_document_last")
                if text and text.strip():
                    # Save raw-ish text for sampling (Pass 1)
                    # We do minimal cleaning here so sampling is consistent with processing
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
