"""
Split cleaned bill documents into paragraph-sized chunks.

Bills are split using section headings and newline runs as boundaries.
Very short chunks are merged with neighbors to ensure meaningful units.
"""
import csv
import os
import re
from pathlib import Path

import pandas as pd

# Regex patterns for section headings
SECTION_RE = re.compile(
    r'(?:\n|\A)(?:\s*)(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?)'
    r'(?=\s)',
    re.IGNORECASE,
)
SECTION_ANY_RE = re.compile(
    r'\b(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?)',
    re.IGNORECASE,
)


def decide_newline_threshold(text):
    """Decide the minimum number of consecutive newlines to use as paragraph breaks."""
    lines = [l for l in text.split("\n") if l.strip()]
    non_empty = len(lines)
    double_runs = text.count("\n\n")

    if double_runs == 0:
        return 1
    if abs(non_empty - double_runs) <= 2:
        return 3
    return 2


def split_into_chunks(text):
    """Split a bill document into paragraph-like chunks."""
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    # Case 1: No newlines at all - use section headings as boundaries
    if "\n" not in text:
        boundaries = []
        for m in SECTION_ANY_RE.finditer(text):
            if m.start() > 0:
                boundaries.append(m.start())
        boundaries = sorted(set(b for b in boundaries if 0 < b < len(text)))
        starts = [0] + boundaries
        chunks = []
        for i in range(len(starts)):
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return merge_short_chunks(chunks)

    # Case 2: Document contains newlines
    # Normalize heavy use of double newlines
    double_runs = text.count("\n\n")
    isolated_runs = len(re.findall(r'(?<!\n)\n(?!\n)', text))
    if double_runs > 0 and double_runs >= 2 * isolated_runs:
        text = re.sub(r'(?<!\n)\n\n(?!\n)', '\n', text)

    threshold = decide_newline_threshold(text)

    # Collect section boundaries (only at start of lines)
    boundaries = []
    for m in SECTION_RE.finditer(text):
        boundaries.append(m.start())
        
    # Collect newline run boundaries
    newline_pat = re.compile(r'\n{' + str(threshold) + ',}')
    for m in newline_pat.finditer(text):
        boundaries.append(m.start())
    boundaries = sorted(set(b for b in boundaries if 0 < b < len(text)))
   
    starts = [0] + boundaries
    chunks = []
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    # If only one short chunk, try line-level heuristic
    if len(chunks) <= 1 and len(chunks[0]) < 250:
        lines = text.split("\n")
        cur = []
        heur = []
        for line in lines:
            if not line.strip():
                if cur:
                    heur.append("\n".join(cur).strip())
                    cur = []
                continue
            cur.append(line)
            if len(line.split()) <= 5:
                heur.append("\n".join(cur).strip())
                cur = []
        if cur:
            heur.append("\n".join(cur).strip())
        if len(heur) > len(chunks):
            chunks = [c for c in heur if c]

    return merge_short_chunks(chunks)


def merge_short_chunks(chunks):
    """Merge chunks shorter than 250 characters with the shorter of their neighbors."""
    if not chunks:
        return []

    merged = []
    i = 0
    n = len(chunks)

    while i < n:
        c = chunks[i].strip()

        if len(c) >= 250 or n == 1:
            merged.append(c)
            i += 1
            continue

        prev_exists = len(merged) > 0
        next_exists = i + 1 < n

        if prev_exists and next_exists:
            prev_len = len(merged[-1])
            next_len = len(chunks[i + 1].strip())
            if prev_len <= next_len:
                merged[-1] = merged[-1] + "\n" + c
                i += 1
            else:
                c = c + "\n" + chunks[i + 1].strip()
                merged.append(c)
                i += 2
        elif prev_exists:
            merged[-1] = merged[-1] + "\n" + c
            i += 1
        elif next_exists:
            c = c + "\n" + chunks[i + 1].strip()
            merged.append(c)
            i += 2
        else:
            merged.append(c)
            i += 1

    return [c for c in merged if c.strip()]


def chunk_dataframe(df):
    """
    Expand a DataFrame of cleaned bills into chunked rows.
    
    Returns a new DataFrame where each row is a chunk with a chunk_id.
    Skips documents with bill_version == "first".
    """
    chunk_rows = []
    for _, row in df.iterrows():
        if row.get('bill_version') == "first":
            continue

        text = row.get("bill_text", "") or ""
        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks, start=1):
            record = {
                'state': row.get('state'),
                'session': row.get('session'),
                'bill_id': row.get('bill_id'),
                'unique_id': row.get('unique_id'),
                'sunlight_id': row.get('sunlight_id'),
                'bill_version': row.get('bill_version'),
                'chunk_id': f"chunk_{idx}",
                'bill_text': chunk,
            }
            chunk_rows.append(record)

    return pd.DataFrame(chunk_rows)


def chunk_csv(input_csv, output_csv, chunksize=1000):
    """
    Stream chunking from a cleaned CSV to an output CSV to avoid memory issues.
    
    Returns per-state stats: { state: { 'documents': int, 'chunks': int, 'words': int } }
    """
    input_csv = str(input_csv)
    output_csv = str(output_csv)

    if os.path.exists(output_csv):
        os.remove(output_csv)

    first_write = True
    stats = {}
    wrote_any = False

    for df_clean in pd.read_csv(input_csv, chunksize=chunksize):
        df_chunks = chunk_dataframe(df_clean)
        if not df_chunks.empty:
            df_chunks.to_csv(output_csv, index=False, mode='a', header=first_write)
            first_write = False
            wrote_any = True

            for state, grp in df_chunks.groupby('state'):
                docs = grp[['bill_id', 'bill_version']].drop_duplicates().shape[0]
                chunks = len(grp)
                words = int(grp['bill_text'].apply(lambda x: len(str(x).split())).sum())

                cur = stats.get(state, {'documents': 0, 'chunks': 0, 'words': 0})
                cur['documents'] += docs
                cur['chunks'] += chunks
                cur['words'] += words
                stats[state] = cur

    if not wrote_any:
        empty_cols = ['state', 'session', 'bill_id', 'unique_id', 'sunlight_id', 
                      'bill_version', 'chunk_id', 'bill_text']
        pd.DataFrame(columns=empty_cols).to_csv(output_csv, index=False)

    return stats


def print_chunk_stats(df_chunks):
    """Print chunk statistics from a DataFrame of chunks."""
    header = [
        'state',
        'documents',
        'chunks',
        'mean_chunks_per_doc',
        'mean_words_per_chunk',
    ]
    rows = []
    total_docs = 0
    total_chunks = 0
    total_words = 0

    for state, grp in df_chunks.groupby('state'):
        docs = grp[['bill_id', 'bill_version']].drop_duplicates().shape[0]
        chunks = len(grp)
        word_counts = grp['bill_text'].apply(lambda x: len(str(x).split()))
        mean_words = float(word_counts.mean()) if len(word_counts) else 0.0
        mean_chunks = (chunks / docs) if docs else 0.0

        total_docs += docs
        total_chunks += chunks
        total_words += int(word_counts.sum())

        rows.append({
            'state': state,
            'documents': docs,
            'chunks': chunks,
            'mean_chunks_per_doc': mean_chunks,
            'mean_words_per_chunk': mean_words,
        })

    overall_mean_chunks = (total_chunks / total_docs) if total_docs else 0.0
    overall_mean_words = (total_words / total_chunks) if total_chunks else 0.0

    rows.append({
        'state': 'TOTAL',
        'documents': total_docs,
        'chunks': total_chunks,
        'mean_chunks_per_doc': overall_mean_chunks,
        'mean_words_per_chunk': overall_mean_words,
    })

    df = pd.DataFrame(rows)[header]
    df['mean_chunks_per_doc'] = df['mean_chunks_per_doc'].round(2)
    df['mean_words_per_chunk'] = df['mean_words_per_chunk'].round(2)
    print(df.to_string(index=False))


def print_chunk_stats_summary(stats):
    """Print chunk stats from the aggregated dict returned by chunk_csv."""
    header = [
        'state',
        'documents',
        'chunks',
        'mean_chunks_per_doc',
        'mean_words_per_chunk',
    ]
    rows = []
    total_docs = 0
    total_chunks = 0
    total_words = 0

    for state in sorted(stats.keys()):
        s = stats[state]
        docs = s['documents']
        chunks = s['chunks']
        words = s['words']
        mean_chunks = (chunks / docs) if docs else 0.0
        mean_words = (words / chunks) if chunks else 0.0

        total_docs += docs
        total_chunks += chunks
        total_words += words

        rows.append({
            'state': state,
            'documents': docs,
            'chunks': chunks,
            'mean_chunks_per_doc': mean_chunks,
            'mean_words_per_chunk': mean_words,
        })

    overall_mean_chunks = (total_chunks / total_docs) if total_docs else 0.0
    overall_mean_words = (total_words / total_chunks) if total_chunks else 0.0

    rows.append({
        'state': 'TOTAL',
        'documents': total_docs,
        'chunks': total_chunks,
        'mean_chunks_per_doc': overall_mean_chunks,
        'mean_words_per_chunk': overall_mean_words,
    })

    df = pd.DataFrame(rows)[header]
    df['mean_chunks_per_doc'] = df['mean_chunks_per_doc'].round(2)
    df['mean_words_per_chunk'] = df['mean_words_per_chunk'].round(2)
    print(df.to_string(index=False))
