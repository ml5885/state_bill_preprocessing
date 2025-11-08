"""
Split cleaned bill documents into paragraph-sized chunks.

Bills are split using section headings and newline runs as boundaries.
Very short chunks are merged with neighbors to ensure meaningful units.
"""
import csv
import os
import random
import re
from pathlib import Path

import pandas as pd

# Minimum chunk length in characters
MIN_CHUNK_LENGTH = 250

# Regex patterns for section headings
SECTION_RE = re.compile(
    r'(?:\n|\A)(?:\s*)(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?)'
    r'(?=\s)',
)
SECTION_ANY_RE = re.compile(
    r'\b(Section\s+\d+[A-Za-z]?\.?|Sec\.\s*\d+[A-Za-z]?\.?|SECTION\s+\d+[A-Za-z]?\.?)',
)

SUBSECTION_RE = re.compile(
    r'(?:\n|\A)\s*'
    r'(\([a-z]\)|\([ivxlcdm]+\)|\(\d+\))'
    r'(?=\s)',
    re.IGNORECASE
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
    
    # Collect subsection boundaries (e.g., (a), (i), (1))
    for m in SUBSECTION_RE.finditer(text):
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

    return merge_short_chunks(chunks)


def extract_section_number(text):
    """Extract the section number from text that starts with a section header."""
    match = SECTION_ANY_RE.match(text.strip())
    if not match:
        return None
    
    # Extract just the number from patterns like "Section 5", "Sec. 10", "SECTION 3A"
    section_text = match.group(1)
    # Find the numeric part
    num_match = re.search(r'(\d+)', section_text)
    if num_match:
        return int(num_match.group(1))
    return None


def is_bounded_section(chunk, prev_chunk=None, next_chunk=None):
    """
    Check if a chunk is a clearly marked section that should not be merged.
    
    A chunk is protected if:
    - It starts with a section header (SECTION X, Sec. X, etc.)
    - The next chunk also starts with a section header with an incrementing number
      (or there is no next chunk)
    """
    # Check if this chunk starts with a section header
    chunk_stripped = chunk.strip()
    current_num = extract_section_number(chunk_stripped)
    
    if current_num is None:
        return False
    
    # If there's a next chunk, check if it also starts with a section header
    # and has a higher section number (incrementing)
    if next_chunk is not None:
        next_stripped = next_chunk.strip()
        next_num = extract_section_number(next_stripped)
        
        # Both have section numbers - check if incrementing
        if next_num is not None:
            # Allow incrementing by 1 or more (e.g., Section 5 -> Section 6 or Section 7)
            if next_num > current_num:
                return True
        
        # Next chunk doesn't have a section number - not a bounded section
        return False
    
    # If this is the last chunk but starts with a section header, also protect it
    return True


def merge_short_chunks(chunks):
    """Merge chunks shorter than MIN_CHUNK_LENGTH characters with neighbors iteratively."""
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
            
            # Check if this is a bounded section that should not be merged
            prev_chunk = merged[-1] if len(merged) > 0 else None
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            
            if is_bounded_section(c, prev_chunk, next_chunk):
                # This is a clearly marked section - keep it as is
                merged.append(c)
                i += 1
                continue
            
            # Chunk is too short - try to merge with a neighbor
            can_merge_prev = len(merged) > 0
            can_merge_next = i + 1 < len(chunks)
            
            if can_merge_prev and can_merge_next:
                # Choose shorter neighbor to merge with
                prev_len = len(merged[-1])
                next_len = len(chunks[i + 1])
                
                if prev_len <= next_len:
                    # Merge with previous
                    merged[-1] = merged[-1] + "\n\n" + c
                    any_merged = True
                    i += 1
                else:
                    # Merge with next
                    merged.append(c + "\n\n" + chunks[i + 1])
                    any_merged = True
                    i += 2
                    
            elif can_merge_prev:
                # Only previous exists - merge with it
                merged[-1] = merged[-1] + "\n\n" + c
                any_merged = True
                i += 1
                
            elif can_merge_next:
                # Only next exists - merge with it
                merged.append(c + "\n\n" + chunks[i + 1])
                any_merged = True
                i += 2
                
            else:
                # No neighbors to merge with - keep the short chunk
                merged.append(c)
                i += 1
        
        chunks = merged
        
        # If no merges happened this iteration, we're done
        if not any_merged:
            break
    
    return chunks


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


def chunk_csv(input_csv, output_csv, chunksize=1000, spot_check_dir=None):
    """
    Stream chunking from a cleaned CSV to an output CSV to avoid memory issues.
    
    Args:
        input_csv: Path to cleaned bills CSV
        output_csv: Path to write chunks CSV
        chunksize: Number of rows to process at a time
        spot_check_dir: Optional directory to save random bill samples per state
    
    Returns per-state stats: { state: { 'documents': int, 'chunks': int, 'words': int } }
    """
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

    # Save random bill samples if spot_check_dir is provided
    if spot_check_dir and wrote_any:
        save_random_bill_samples(output_csv, spot_check_dir)

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
    df.to_csv('chunk_stats_summary.csv', index=False)


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
            word_count = len(text.split())
            
            print(f"\nExample {idx}:")
            print(f"  Bill ID: {bill_id}")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  Word Count: {word_count}")
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
        
        total_words = sum(len(str(chunk).split()) for chunk in bill_chunks['bill_text'])
        f.write(f"Total Words: {total_words}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CHUNKS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, (_, row) in enumerate(bill_chunks.iterrows(), start=1):
            chunk_id = row['chunk_id']
            chunk_text = str(row['bill_text'])
            chunk_words = len(chunk_text.split())
            
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Chunk {idx} (ID: {chunk_id})\n")
            f.write(f"Word Count: {chunk_words}\n")
            f.write(f"{'-' * 80}\n\n")
            f.write(chunk_text)
            f.write("\n")

