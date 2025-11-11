"""
Split cleaned bill documents into paragraph-sized chunks.

Bills are split using section headings and newline runs as boundaries.
Very short chunks are merged with neighbors to ensure meaningful units.
"""
import csv
import os
import random
import re
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Minimum chunk length in characters
MIN_CHUNK_LENGTH = 300

# Maximum chunk length in characters - chunks longer than this may be split
MAX_CHUNK_LENGTH = 1000

# Regex patterns for section headings
# Pattern for section headings at start of line or after newline
SECTION_RE = re.compile(
    r'(?:\n|\A)(?:\s*)(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SEC\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?|SECTION\s+AUTONUMLGL\s+\\e\s*\.?|S\s+\d+\.)'
    r'(?=\s)',
)

# Pattern for section headings anywhere in text (for no-newline case)
SECTION_ANY_RE = re.compile(
    r'(?:\b|(?<=\W))(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SEC\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?|SECTION\s+AUTONUMLGL\s+\\e\s*\.?|S\s+\d+\.)',
)

# Pattern for subsections: (a), (1), (A), (i), etc.
# Match at start of line (after newline or at beginning) OR after punctuation like : or ;
SUBSECTION_RE = re.compile(
    r'(?:^|\n|(?<=:)\s+|(?<=;)\s+)(\s*)(\([a-zA-Z]\)|\([ivxlcdm]+\)|\(\d+\))',
    re.IGNORECASE | re.MULTILINE
)

# Pattern for numeric subsections: "1.", "2.", "A.", "B.", etc.
# Match at start of line (after newline) OR after punctuation like : or ;
NUMERIC_SUBSECTION_RE = re.compile(
    r'(?:^|\n|(?<=:)\s+|(?<=;)\s+)(\s*)(\d+\.|[A-Z]\.)',
    re.MULTILINE
)

# Pattern for subsections in no-newline text (more relaxed)
# Matches (a), (1), (A), etc. after periods, colons, semicolons, or closing brackets
SUBSECTION_NO_NEWLINE_RE = re.compile(
    r'(?:[\.;:\])\s+)(\([a-zA-Z]\)|\([ivxlcdm]+\)|\(\d+\))',
    re.IGNORECASE
)

# Pattern for paragraph breaks: newline followed by tabs/spaces (common in legal text)
# This pattern like '\n\t\t\t  ' indicates a paragraph break
PARAGRAPH_BREAK_RE = re.compile(r'\n\t+\s*')


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

    has_newlines = "\n" in text
    has_double_newlines = "\n\n" in text
    
    if not has_newlines:
        boundaries = []
        
        # Find Section boundaries
        for m in SECTION_ANY_RE.finditer(text):
            if m.start() > 0:
                boundaries.append(m.start())
        
        # Find subsection boundaries like (a), (1), etc.
        # Use the relaxed pattern for no-newline text
        for m in SUBSECTION_NO_NEWLINE_RE.finditer(text):
            # The pattern captures the subsection marker in group 1
            # We want the boundary at the start of the '(' character
            match_text = text[m.start():m.end()]
            paren_pos = match_text.find('(')
            if paren_pos >= 0:
                boundary_pos = m.start() + paren_pos
                if boundary_pos > 0:
                    boundaries.append(boundary_pos)
        
        # Find numeric subsections like "1.", "A.", etc.
        for m in NUMERIC_SUBSECTION_RE.finditer(text):
            match_text = text[m.start():m.end()]
            for i, ch in enumerate(match_text):
                if ch.isdigit() or ch in string.ascii_uppercase:
                    boundary_pos = m.start() + i
                    if boundary_pos > 0:
                        boundaries.append(boundary_pos)
                    break
        
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

    double_runs = text.count("\n\n")
    isolated_runs = len(re.findall(r'(?<!\n)\n(?!\n)', text))
    if double_runs > 0 and double_runs >= 2 * isolated_runs:
        text = re.sub(r'(?<!\n)\n\n(?!\n)', '\n', text)

    threshold = decide_newline_threshold(text)
    boundaries = []
    
    if not has_double_newlines:
        section_relaxed_re = re.compile(
            r'(?:(?<=\s)|(?<=\W)|(?<=\.))'
            r'(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SEC\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?|SECTION\s+AUTONUMLGL\s+\\e\s*\.?|S\s+\d+\.)'
            r'(?=\s)',
        )
        for m in section_relaxed_re.finditer(text):
            if m.start() > 0:
                boundaries.append(m.start())
    else:
        for m in SECTION_RE.finditer(text):
            boundaries.append(m.start())
    
    for m in SUBSECTION_RE.finditer(text):
        match_text = text[m.start():m.end()]
        paren_pos = match_text.find('(')
        if paren_pos >= 0:
            boundary_pos = m.start() + paren_pos
            if boundary_pos > 0:
                boundaries.append(boundary_pos)
    
    for m in NUMERIC_SUBSECTION_RE.finditer(text):
        match_text = text[m.start():m.end()]
        for i, ch in enumerate(match_text):
            if ch.isdigit() or ch in string.ascii_uppercase:
                boundary_pos = m.start() + i
                if boundary_pos > 0:
                    boundaries.append(boundary_pos)
                break
    
    # Add paragraph break boundaries (newline + tabs/spaces pattern)
    for m in PARAGRAPH_BREAK_RE.finditer(text):
        if m.start() > 0:
            boundaries.append(m.start())
        
    if threshold > 1:
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

    if len(chunks) <= 1 and len(chunks[0]) < 500:
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

    result = merge_short_chunks(chunks)
    
    # Split any chunks that are too long
    total_bill_length = len(text)
    result = split_long_chunks(result, total_bill_length)
    
    return result


def extract_section_number(text):
    """Extract the section number from text that starts with a section header."""
    match = SECTION_ANY_RE.match(text.strip())
    if not match:
        return None
    
    section_text = match.group(1)
    num_match = re.search(r'(\d+)', section_text)
    if num_match:
        return int(num_match.group(1))
    return None


def validate_consecutive_sections(chunks):
    """Validate that SEC./Section boundaries are consecutive."""
    if not chunks:
        return set()
    
    section_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_stripped = chunk.strip()
        sec_num = extract_section_number(chunk_stripped)
        if sec_num is not None:
            section_chunks.append((idx, sec_num))
    
    if len(section_chunks) < 2:
        return set(idx for idx, _ in section_chunks)
    
    protected = set()
    for i in range(len(section_chunks)):
        curr_idx, curr_num = section_chunks[i]
        
        if i + 1 < len(section_chunks):
            next_idx, next_num = section_chunks[i + 1]
            if next_num > curr_num:
                protected.add(curr_idx)
                if i == len(section_chunks) - 2:
                    protected.add(next_idx)
        else:
            if i > 0:
                prev_idx, prev_num = section_chunks[i - 1]
                if curr_num > prev_num:
                    protected.add(curr_idx)
    
    return protected



def is_bounded_section(chunk, prev_chunk=None, next_chunk=None):
    """Check if a chunk should be protected from merging."""
    chunk_stripped = chunk.strip()
    current_num = extract_section_number(chunk_stripped)
    
    if current_num is not None:
        if len(chunk_stripped) < MIN_CHUNK_LENGTH * 0.4: 
            return False
            
        if next_chunk is not None:
            next_stripped = next_chunk.strip()
            next_num = extract_section_number(next_stripped)
            if next_num is not None and next_num > current_num:
                return True
            return False
        
        return True
    
    subsection_start_re = re.compile(r'^(\([a-zA-Z0-9ivxlcdm]+\)|\d+\.|[A-Z]\.)\s+', re.IGNORECASE)
    if subsection_start_re.match(chunk_stripped):
        if len(chunk_stripped) < MIN_CHUNK_LENGTH * 0.6:
            return False
        
        if next_chunk is not None:
            next_stripped = next_chunk.strip()
            if subsection_start_re.match(next_stripped):
                return True
        
        return False
    
    return False


def merge_short_chunks(chunks):
    """Merge chunks shorter than MIN_CHUNK_LENGTH characters with neighbors iteratively.
    
    Algorithm illustration (assuming MIN_CHUNK_LENGTH=500):
    
    Initial state:
        [100 chars] [200 chars] [50 chars] [600 chars] [300 chars] [80 chars]
         chunk1      chunk2      chunk3     chunk4      chunk5      chunk6
    
    Iteration 1:
        - chunk1 (100) < 500: merge with next -> [300] (merged chunk1+chunk2)
        - chunk3 (50) < 500: merge with next -> [650] (merged chunk3+chunk4)
        - chunk5 (300) < 500: merge with next -> [380] (merged chunk5+chunk6)

        Result: [[300], [650], [380]]
                 ^merge  ^keep   ^merge
    
    Iteration 2:
        - [300] < 500: merge with next -> [950] (merged [300]+[650])
        - [380] < 500: stays as-is (no next chunk available)
        
        Result: [[950], [380]]
                 ^keep  ^short
    
    Iteration 3:
        - No more merges possible (would require merging [380] backward into [950])
        
    Final: [[950], [380]]
    
    Special cases:
        - Protected sections (consecutive SEC. N headers) are never merged
        - When choosing merge direction, prefer forward if result <= MIN_CHUNK_LENGTH
        - Otherwise prefer smaller total size (up to 2*MIN_CHUNK_LENGTH)
    """
    if not chunks:
        return []
    
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return chunks

    # Keep merging until all chunks are >= MIN_CHUNK_LENGTH (or we can't merge anymore)
    max_iterations = len(chunks)
    for iteration in range(max_iterations):
        merged = []
        i = 0
        any_merged = False
        
        while i < len(chunks):
            c = chunks[i]
            
            # If this chunk is long enough, keep it
            if len(c) >= MIN_CHUNK_LENGTH:
                merged.append(c)
                i += 1
                continue
            
            prev_chunk = merged[-1] if len(merged) > 0 else None
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            
            if is_bounded_section(c, prev_chunk, next_chunk):
                merged.append(c)
                i += 1
                continue
            
            can_merge_prev = len(merged) > 0
            can_merge_next = i + 1 < len(chunks)
            
            if can_merge_prev and can_merge_next:
                prev_len = len(merged[-1])
                next_len = len(chunks[i + 1])
                curr_len = len(c)
                
                merge_prev_size = prev_len + curr_len
                merge_next_size = curr_len + next_len
                
                if merge_next_size <= MIN_CHUNK_LENGTH or (merge_next_size < merge_prev_size and merge_next_size <= 2 * MIN_CHUNK_LENGTH):
                    merged.append(c + "\n\n" + chunks[i + 1])
                    any_merged = True
                    i += 2
                else:
                    merged[-1] = merged[-1] + "\n\n" + c
                    any_merged = True
                    i += 1
                    
            elif can_merge_prev:
                merged[-1] = merged[-1] + "\n\n" + c
                any_merged = True
                i += 1
                
            elif can_merge_next:
                merged.append(c + "\n\n" + chunks[i + 1])
                any_merged = True
                i += 2
                
            else:
                merged.append(c)
                i += 1
        
        chunks = merged
        
        if not any_merged:
            break
    
    return chunks


def split_long_chunks(chunks, total_bill_length):
    """Split chunks that are too long based on certain conditions.
    
    A chunk is split if:
    1. It's longer than MAX_CHUNK_LENGTH AND (there are fewer than 3 total chunks 
       OR the chunk is >40% of total bill length)
    2. The chunk contains newline characters
    
    When splitting:
    - First try to split on subsection boundaries (most common structure)
    - Then try section boundaries
    - Otherwise split on newline characters
    - After splitting, merge any resulting chunks that are too small
    """
    if not chunks:
        return []
    
    result = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        
        # Check if this chunk should be split
        should_split = False
        
        # Condition 1: chunk is too long AND (few total chunks OR chunk is large portion of bill)
        if chunk_len > MAX_CHUNK_LENGTH:
            if len(chunks) < 3:
                should_split = True
            elif total_bill_length > 0 and chunk_len > 0.4 * total_bill_length:
                should_split = True
        
        # Condition 2: chunk contains newlines and is too long
        has_newlines = '\n' in chunk
        if chunk_len > MAX_CHUNK_LENGTH and has_newlines:
            should_split = True
        
        if not should_split:
            result.append(chunk)
            continue
        
        # Try to split the chunk
        # First, try to find subsection boundaries (most common in legal text)
        boundaries = []
        
        # Look for paragraph break patterns (newline + tabs)
        for m in PARAGRAPH_BREAK_RE.finditer(chunk):
            if m.start() > 0:
                boundaries.append(m.start())
        
        # Look for subsection boundaries like (a), (b), (c)
        for m in SUBSECTION_RE.finditer(chunk):
            match_text = chunk[m.start():m.end()]
            paren_pos = match_text.find('(')
            if paren_pos >= 0:
                boundary_pos = m.start() + paren_pos
                if boundary_pos > 0:
                    boundaries.append(boundary_pos)
        
        # Also check numeric subsections like "1.", "2.", "A.", "B."
        if not boundaries:
            for m in NUMERIC_SUBSECTION_RE.finditer(chunk):
                match_text = chunk[m.start():m.end()]
                for i, ch in enumerate(match_text):
                    if ch.isdigit() or ch in string.ascii_uppercase:
                        boundary_pos = m.start() + i
                        if boundary_pos > 0:
                            boundaries.append(boundary_pos)
                        break
        
        # If no subsection boundaries, look for section boundaries
        if not boundaries:
            for m in SECTION_RE.finditer(chunk):
                if m.start() > 0:  # Don't split at the very beginning
                    boundaries.append(m.start())
        
        # If we found section/subsection boundaries, use them
        if boundaries:
            boundaries = sorted(set(boundaries))
            
            starts = [0] + boundaries
            sub_chunks = []
            for i in range(len(starts)):
                start = starts[i]
                end = starts[i + 1] if i + 1 < len(starts) else len(chunk)
                sub_chunk = chunk[start:end].strip()
                if sub_chunk:
                    sub_chunks.append(sub_chunk)
            
            # Merge sub-chunks that are too small
            merged_sub_chunks = merge_short_chunks(sub_chunks)
            result.extend(merged_sub_chunks)
        
        # Otherwise, if chunk has newlines, split on them
        elif has_newlines:
            # Decide on newline threshold
            threshold = decide_newline_threshold(chunk)
            if threshold > 1:
                newline_pat = re.compile(r'\n{' + str(threshold) + ',}')
                newline_boundaries = []
                for m in newline_pat.finditer(chunk):
                    if m.start() > 0:
                        newline_boundaries.append(m.start())
                
                if newline_boundaries:
                    newline_boundaries = sorted(set(newline_boundaries))
                    starts = [0] + newline_boundaries
                    sub_chunks = []
                    for i in range(len(starts)):
                        start = starts[i]
                        end = starts[i + 1] if i + 1 < len(starts) else len(chunk)
                        sub_chunk = chunk[start:end].strip()
                        if sub_chunk:
                            sub_chunks.append(sub_chunk)
                    
                    # Merge sub-chunks that are too small
                    merged_sub_chunks = merge_short_chunks(sub_chunks)
                    result.extend(merged_sub_chunks)
                else:
                    # Fallback: split on any newline
                    sub_chunks = [s.strip() for s in chunk.split('\n') if s.strip()]
                    merged_sub_chunks = merge_short_chunks(sub_chunks)
                    result.extend(merged_sub_chunks)
            else:
                # Split on single newlines
                sub_chunks = [s.strip() for s in chunk.split('\n') if s.strip()]
                merged_sub_chunks = merge_short_chunks(sub_chunks)
                result.extend(merged_sub_chunks)
        else:
            # Can't split, keep as-is
            result.append(chunk)
    
    return result


def chunk_dataframe(df):
    """Expand a DataFrame of cleaned bills into chunked rows."""
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


def chunk_csv(input_csv, output_csv, chunksize=1000, spot_check_dir=None):
    """Stream chunking from a cleaned CSV to an output CSV."""
    input_csv = str(input_csv)
    output_csv = str(output_csv)

    if os.path.exists(output_csv):
        os.remove(output_csv)

    first_write = True
    stats = {}
    wrote_any = False

    for df_clean in pd.read_csv(input_csv, chunksize=chunksize, dtype={'session': str}):
        df_chunks = chunk_dataframe(df_clean)
        if not df_chunks.empty:
            df_chunks.to_csv(output_csv, index=False, mode='a', header=first_write)
            first_write = False
            wrote_any = True

            for state, grp in df_chunks.groupby('state'):
                docs = grp[['bill_id', 'bill_version']].drop_duplicates().shape[0]
                chunks = len(grp)
                char_counts_list = grp['bill_text'].apply(lambda x: len(str(x))).tolist()
                chars = sum(char_counts_list)
                word_counts_list = grp['bill_text'].apply(lambda x: len(str(x).split())).tolist()
                words = sum(word_counts_list)

                cur = stats.get(state, {'documents': 0, 'chunks': 0, 'chars': 0, 'char_counts': [], 'words': 0, 'word_counts': []})
                cur['documents'] += docs
                cur['chunks'] += chunks
                cur['chars'] += chars
                cur['char_counts'].extend(char_counts_list)
                cur['words'] += words
                cur['word_counts'].extend(word_counts_list)
                stats[state] = cur

    if not wrote_any:
        empty_cols = ['state', 'session', 'bill_id', 'unique_id', 'sunlight_id', 
                      'bill_version', 'chunk_id', 'bill_text']
        pd.DataFrame(columns=empty_cols).to_csv(output_csv, index=False)

    # Save random bill samples if spot_check_dir is provided
    if spot_check_dir and wrote_any:
        save_random_bill_samples(output_csv, spot_check_dir)

    return stats


def print_chunk_stats(data, save_stats=True):
    """Print chunk statistics using character counts instead of words."""
    header = [
        'state',
        'documents',
        'chunks',
        'mean_chunks_per_doc',
        'mean_chars_per_chunk',
        'mean_words_per_chunk',
        'min_chars',
        'q25_chars',
        'median_chars',
        'q75_chars',
        'max_chars',
    ]
    rows = []
    total_docs = 0
    total_chunks = 0
    total_chars = 0
    total_words = 0
    all_char_counts = []

    if isinstance(data, pd.DataFrame):
        pbar = tqdm(total=len(data['state'].unique()), desc="Computing chunk stats")
        for state, grp in data.groupby('state'):
            docs = grp[['bill_id', 'bill_version']].drop_duplicates().shape[0]
            chunks = len(grp)
            char_counts = grp['bill_text'].apply(lambda x: len(str(x)))
            char_counts_list = char_counts.tolist()
            mean_chars = float(char_counts.mean()) if len(char_counts) else 0.0
            word_counts = grp['bill_text'].apply(lambda x: len(str(x).split()))
            mean_words = float(word_counts.mean()) if len(word_counts) else 0.0
            mean_chunks = (chunks / docs) if docs else 0.0

            total_docs += docs
            total_chunks += chunks
            total_chars += int(char_counts.sum())
            total_words += int(word_counts.sum())
            all_char_counts.extend(char_counts_list)

            if char_counts_list:
                min_chars = int(np.min(char_counts_list))
                q25_chars = int(np.percentile(char_counts_list, 25))
                median_chars = int(np.median(char_counts_list))
                q75_chars = int(np.percentile(char_counts_list, 75))
                max_chars = int(np.max(char_counts_list))
            else:
                min_chars = q25_chars = median_chars = q75_chars = max_chars = 0

            rows.append({
                'state': state,
                'documents': docs,
                'chunks': chunks,
                'mean_chunks_per_doc': mean_chunks,
                'mean_chars_per_chunk': mean_chars,
                'mean_words_per_chunk': mean_words,
                'min_chars': min_chars,
                'q25_chars': q25_chars,
                'median_chars': median_chars,
                'q75_chars': q75_chars,
                'max_chars': max_chars,
            })
            pbar.update(1)
        pbar.close()
    else:
        for state in sorted(data.keys()):
            s = data[state]
            docs = s['documents']
            chunks = s['chunks']
            # Prefer character counts if present; also include mean words if available
            if 'chars' in s and 'char_counts' in s:
                chars = s['chars']
                char_counts = s.get('char_counts', [])
                mean_chars = (chars / chunks) if chunks else 0.0
                # Mean words from aggregated words/word_counts when present
                if 'words' in s and s.get('words', 0) and chunks:
                    mean_words = (s['words'] / chunks)
                elif 'word_counts' in s and s.get('word_counts'):
                    wc = s['word_counts']
                    mean_words = (sum(wc) / len(wc)) if len(wc) else 0.0
                else:
                    mean_words = 0.0
                if char_counts:
                    min_chars = int(np.min(char_counts))
                    q25_chars = int(np.percentile(char_counts, 25))
                    median_chars = int(np.median(char_counts))
                    q75_chars = int(np.percentile(char_counts, 75))
                    max_chars = int(np.max(char_counts))
                else:
                    min_chars = q25_chars = median_chars = q75_chars = max_chars = 0

                total_docs += docs
                total_chunks += chunks
                total_chars += chars
                total_words += int(s.get('words', 0))
                all_char_counts.extend(char_counts)
            else:
                # Fallback for older stats structures (word-based)
                words = s.get('words', 0)
                word_counts = s.get('word_counts', [])
                mean_chars = 0.0
                # We don't have char distribution; we can't compute min/q stats; keep zeros.
                mean_words = (words / chunks) if chunks else 0.0
                if word_counts:
                    # No char counts; leave char percentiles as 0 for compatibility
                    pass
                else:
                    min_chars = q25_chars = median_chars = q75_chars = max_chars = 0

                total_docs += docs
                total_chunks += chunks
                total_words += words

            rows.append({
                'state': state,
                'documents': docs,
                'chunks': chunks,
                'mean_chunks_per_doc': (chunks / docs) if docs else 0.0,
                'mean_chars_per_chunk': mean_chars,
                'mean_words_per_chunk': mean_words,
                'min_chars': min_chars,
                'q25_chars': q25_chars,
                'median_chars': median_chars,
                'q75_chars': q75_chars,
                'max_chars': max_chars,
            })

    overall_mean_chunks = (total_chunks / total_docs) if total_docs else 0.0
    overall_mean_chars = (total_chars / total_chunks) if total_chunks else 0.0
    overall_mean_words = (total_words / total_chunks) if total_chunks else 0.0

    if all_char_counts:
        overall_min = int(np.min(all_char_counts))
        overall_q25 = int(np.percentile(all_char_counts, 25))
        overall_median = int(np.median(all_char_counts))
        overall_q75 = int(np.percentile(all_char_counts, 75))
        overall_max = int(np.max(all_char_counts))
    else:
        overall_min = overall_q25 = overall_median = overall_q75 = overall_max = 0

    rows.append({
        'state': 'TOTAL',
        'documents': total_docs,
        'chunks': total_chunks,
        'mean_chunks_per_doc': overall_mean_chunks,
        'mean_chars_per_chunk': overall_mean_chars,
        'mean_words_per_chunk': overall_mean_words,
        'min_chars': overall_min,
        'q25_chars': overall_q25,
        'median_chars': overall_median,
        'q75_chars': overall_q75,
        'max_chars': overall_max,
    })

    df = pd.DataFrame(rows)[header]
    df['mean_chunks_per_doc'] = df['mean_chunks_per_doc'].round(2)
    df['mean_chars_per_chunk'] = df['mean_chars_per_chunk'].round(2)
    df['mean_words_per_chunk'] = df['mean_words_per_chunk'].round(2)
    print(df.to_string(index=False))
    
    if save_stats:
        df.to_csv('chunk_stats_summary.csv', index=False)

def save_chunk_length_panels(data, output_path='chunk_length_panels.png', bins=200, log_y=True, clip_upper_chars=None, clip_upper_words=None, show_min_threshold=True):
    """Create a side-by-side panel of (char length histogram, word length histogram)."""
    import matplotlib.pyplot as plt

    if isinstance(data, pd.DataFrame):
        char_lengths = data['bill_text'].astype(str).str.len().values
        word_lengths = data['bill_text'].astype(str).apply(lambda s: len(s.split())).values
        total_rows = len(data)
    elif isinstance(data, dict):
        char_acc = []
        word_acc = []
        for s in data.values():
            if isinstance(s, dict):
                if 'char_counts' in s:
                    char_acc.extend(s['char_counts'])
                if 'word_counts' in s:
                    word_acc.extend(s['word_counts'])
        char_lengths = np.array(char_acc, dtype=int)
        word_lengths = np.array(word_acc, dtype=int)
        total_rows = None
    else:
        raise TypeError("Unsupported data type for panels; expected DataFrame or dict.")

    # Filter >0
    char_lengths = char_lengths[char_lengths > 0]
    word_lengths = word_lengths[word_lengths > 0]
    if clip_upper_chars is not None:
        char_lengths = np.clip(char_lengths, 0, clip_upper_chars)
    if clip_upper_words is not None:
        word_lengths = np.clip(word_lengths, 0, clip_upper_words)

    if char_lengths.size == 0 or word_lengths.size == 0:
        print("[WARN] Not enough data to produce panel plot.")
        return

    pct_under_min = float((char_lengths < MIN_CHUNK_LENGTH).sum()) * 100.0 / int(char_lengths.size)
    print(f"[Analysis] Panels: chars={char_lengths.size:,} (median {int(np.median(char_lengths))}, %< {MIN_CHUNK_LENGTH}={pct_under_min:.2f}%), words={word_lengths.size:,} (median {int(np.median(word_lengths))}).")
    if total_rows is not None:
        print(f"[Analysis] Input rows: {total_rows:,} (DataFrame).")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axc, axw = axes

    # Character histogram
    c_counts, c_bins, _ = axc.hist(char_lengths, bins=bins, color="#1f77b4")
    axc.set_title('Chunk Character Lengths')
    axc.set_xlabel('Chars per chunk')
    axc.set_ylabel('Number of chunks')
    if log_y:
        from matplotlib.ticker import LogLocator, FuncFormatter
        axc.set_yscale('log', base=10)
        axc.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        axc.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y):,}" if y >= 1 else ""))
    if show_min_threshold:
        axc.axvline(MIN_CHUNK_LENGTH, color='red', linestyle='--', linewidth=1.2, label=f'{MIN_CHUNK_LENGTH} chars')
        axc.legend()
    axc.grid(alpha=0.2, linestyle=':')

    # Word histogram
    w_counts, w_bins, _ = axw.hist(word_lengths, bins=bins, color="#ff7f0e")
    axw.set_title('Chunk Word Lengths')
    axw.set_xlabel('Words per chunk')
    axw.set_ylabel('Number of chunks')
    if log_y:
        from matplotlib.ticker import LogLocator, FuncFormatter
        axw.set_yscale('log', base=10)
        axw.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        axw.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y):,}" if y >= 1 else ""))
    axw.grid(alpha=0.2, linestyle=':')

    fig.suptitle('Chunk Length Distributions (Characters vs Words)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = str(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    try:
        print(f"[Analysis] Saved combined panels to {output_path} (char max bin {int(c_counts.max()):,}, word max bin {int(w_counts.max()):,}).")
    except Exception:
        print(f"[Analysis] Saved combined panels to {output_path}")

def print_example_chunks(chunks_csv_path, num_examples=3):
    """Print example chunks for each state to verify quality."""
    print("\n" + "=" * 80)
    print("EXAMPLE CHUNKS BY STATE")
    print("=" * 80)
    
    chunks_csv_path = str(chunks_csv_path)
    if not os.path.exists(chunks_csv_path):
        print(f"Chunks file not found: {chunks_csv_path}")
        return
    
    df = pd.read_csv(chunks_csv_path, dtype={'session': str})
    
    for state in sorted(df['state'].unique()):
        state_chunks = df[df['state'] == state]
        sample_size = min(num_examples, len(state_chunks))
        samples = state_chunks.sample(n=sample_size, random_state=42)
        
        print(f"\n{'=' * 80}")
        print(f"STATE: {state.upper()}")
        print(f"{'=' * 80}")
        
        for idx, (_, row) in enumerate(samples.iterrows(), start=1):
            bill_id = row.get('bill_id', 'N/A')
            chunk_id = row.get('chunk_id', 'N/A')
            text = str(row.get('bill_text', ''))
            char_count = len(text)
            
            print(f"\nExample {idx}:")
            print(f"  Bill ID: {bill_id}")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  Char Count: {char_count}")
            print(f"  Text Preview (first 500 chars):")
            print(f"  {'-' * 76}")
            preview = text[:500] + "..." if len(text) > 500 else text
            for line in preview.split('\n'):
                print(f"  {line}")
            print(f"  {'-' * 76}")


def save_random_bill_samples(chunks_csv_path, output_dir="spot_check"):
    chunks_csv_path = str(chunks_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(chunks_csv_path):
        print(f"[WARN] Chunks CSV not found: {chunks_csv_path}")
        return
    
    print(f"\n{'=' * 80}")
    print("SAVING RANDOM BILL SAMPLES FOR SPOT CHECKING")
    print(f"{'=' * 80}")
    
    df = pd.read_csv(chunks_csv_path, dtype={'session': str}, encoding='utf-8', on_bad_lines='skip')
    
    if df.empty:
        print("[WARN] Chunks CSV is empty")
        return
    
    required_cols = {'state', 'unique_id', 'bill_id', 'bill_text', 'chunk_id'}
    if not required_cols.issubset(df.columns):
        print(f"[WARN] Missing required columns. Found: {df.columns.tolist()}")
        return
    
    df = df[df['state'].str.match(r'^[a-z]{2}$', na=False)]
    
    if df.empty:
        print("[WARN] No valid state codes found in chunks CSV")
        return
    
    files_written = 0
    
    for state in sorted(df['state'].unique()):
        state_chunks = df[df['state'] == state].copy()
        unique_bills = state_chunks[['unique_id', 'bill_id']].drop_duplicates()
        
        if unique_bills.empty:
            print(f"{state.upper()}: No bills found")
            continue
        
        sampled_bills = unique_bills.sample(n=1, random_state=random.randint(0, 10000))
        
        for _, bill_row in sampled_bills.iterrows():
            selected_unique_id = bill_row['unique_id']
            selected_bill_id = bill_row['bill_id']
            
            bill_chunks = state_chunks[state_chunks['unique_id'] == selected_unique_id].sort_values('chunk_id')
            
            if bill_chunks.empty:
                continue
            
            safe_bill_id = str(selected_bill_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
            safe_state = str(state).strip().lower()
            
            filename = f"{safe_state}_{safe_bill_id}.txt"
            
            filepath = output_dir / filename
            
            try:
                _write_bill_sample_file(filepath, bill_chunks, safe_state)
                print(f"{safe_state.upper()}: Saved {len(bill_chunks)} chunks from bill {selected_bill_id}")
                files_written += 1
            except Exception as e:
                print(f"{safe_state.upper()}: Error writing file {filename}: {e}")
    
    print(f"\nSaved {files_written} bill samples to {output_dir}/")
    print(f"  Total states: {len(df['state'].unique())}")


def _write_bill_sample_file(filepath, bill_chunks, state_code):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SPOT CHECK: Random Bill Sample for {state_code.upper()}\n")
        f.write("=" * 80 + "\n\n")
        
        first_row = bill_chunks.iloc[0]
        f.write(f"State: {first_row['state']}\n")
        f.write(f"Bill ID: {first_row['bill_id']}\n")
        f.write(f"Unique ID: {first_row['unique_id']}\n")
        f.write(f"Session: {first_row.get('session', 'N/A')}\n")
        f.write(f"Bill Version: {first_row.get('bill_version', 'N/A')}\n")
        f.write(f"Sunlight ID: {first_row.get('sunlight_id', 'N/A')}\n")
        f.write(f"Total Chunks: {len(bill_chunks)}\n")
        total_chars = sum(len(str(chunk)) for chunk in bill_chunks['bill_text'])
        f.write(f"Total Chars: {total_chars}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CHUNKS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, (_, row) in enumerate(bill_chunks.iterrows(), start=1):
            chunk_id = row['chunk_id']
            chunk_text = str(row['bill_text'])
            chunk_chars = len(chunk_text)
            
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Chunk {idx} (ID: {chunk_id})\n")
            f.write(f"Char Count: {chunk_chars}\n")
            f.write(f"{'-' * 80}\n\n")
            f.write(chunk_text)
            f.write("\n")


def save_outlier_chunks(data, output_dir, char_threshold=200000, top_k=50):
    """Save extremely large chunk rows to individual JSON files and an index.

    Args:
        data: DataFrame of chunks (preferred). Must include bill_text and metadata columns.
        output_dir: Directory to write JSON files.
        char_threshold: Minimum character length to treat as an outlier.
        top_k: Always include the top_k longest chunks in addition to threshold filter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(data, pd.DataFrame):
        print("[WARN] save_outlier_chunks expects a DataFrame with bill_text; skipping.")
        return

    df = data.copy()
    df['char_len'] = df['bill_text'].astype(str).str.len()
    df['word_len'] = df['bill_text'].astype(str).apply(lambda s: len(s.split()))

    # Select threshold-based outliers
    mask = df['char_len'] >= int(char_threshold)
    outliers = df[mask]

    # Also include top_k by char length
    if top_k and len(df) > 0:
        topk = df.nlargest(int(top_k), 'char_len')
        outliers = pd.concat([outliers, topk], ignore_index=True).drop_duplicates(subset=['state', 'unique_id', 'bill_version', 'chunk_id'], keep='first')

    if outliers.empty:
        print(f"[Analysis] No outlier chunks found (threshold={char_threshold}).")
        return

    records = []
    for _, row in outliers.sort_values('char_len', ascending=False).iterrows():
        state = str(row.get('state', 'NA')).strip().lower()
        bill_id = str(row.get('bill_id', 'NA'))
        unique_id = str(row.get('unique_id', 'NA'))
        version = str(row.get('bill_version', 'NA'))
        chunk_id = str(row.get('chunk_id', 'NA'))
        session = str(row.get('session', 'NA'))
        sunlight_id = str(row.get('sunlight_id', 'NA'))
        text = str(row.get('bill_text', ''))
        char_len = int(row.get('char_len', 0))
        word_len = int(row.get('word_len', 0))

        safe_bill = bill_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
        safe_chunk = chunk_id.replace('/', '_')
        fname = f"{state}_{safe_bill}_{safe_chunk}_{char_len}.json"
        fpath = output_dir / fname

        payload = {
            'state': state,
            'session': session,
            'bill_id': bill_id,
            'unique_id': unique_id,
            'sunlight_id': sunlight_id,
            'bill_version': version,
            'chunk_id': chunk_id,
            'char_length': char_len,
            'word_length': word_len,
            'bill_text': text,
        }

        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            records.append({k: v for k, v in payload.items() if k != 'bill_text'} | {'file': str(fpath)})
        except Exception as e:
            print(f"[WARN] Failed to write outlier file {fpath.name}: {e}")

    # Write an index summary without full text
    idx_path = output_dir / 'outliers_index.json'
    try:
        with open(idx_path, 'w', encoding='utf-8') as f:
            json.dump({'count': len(records), 'threshold': int(char_threshold), 'top_k': int(top_k), 'items': records}, f, ensure_ascii=False)
        print(f"[Analysis] Saved {len(records)} outlier chunks and index to {output_dir}/")
    except Exception as e:
        print(f"[WARN] Failed to write outliers index: {e}")

