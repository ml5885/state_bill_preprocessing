"""
Augment chunked bills with metadata: date and user_type (state or state_party).

Reads raw bills JSON to extract dates and sponsor party affiliation.
Optionally reads people YAML files to map legislators to parties.
"""
import argparse
import csv
import gzip
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

EXCLUDED_STATES = {'co', 'pr', 'dc'}


def build_legislator_party_map(people_dir):
    """
    Build a mapping from legacy OpenStates legislator IDs to party names.
    
    Returns dict: { legislator_id: party_name }
    """
    leg_map = {}
    if not os.path.isdir(people_dir):
        print(f"[WARN] people_dir not found: {people_dir}")
        return leg_map

    yaml_paths = []
    for root, _, files in os.walk(people_dir):
        segments = root.split(os.sep)
        if 'legislature' not in segments and 'retired' not in segments:
            continue
        for fname in files:
            if fname.lower().endswith(('.yml', '.yaml')):
                yaml_paths.append(os.path.join(root, fname))

    for path in tqdm(yaml_paths, desc="Loading people data", unit="files"):
        data = yaml.safe_load(open(path, 'r', encoding='utf-8'))
        if not isinstance(data, dict):
            continue

        oid = data.get('id')
        if not isinstance(oid, str) or not oid.startswith('ocd-person/'):
            continue

        party_name = 'Unknown'
        parties = data.get('party')
        if isinstance(parties, list) and parties:
            first = parties[0]
            if isinstance(first, dict):
                name = first.get('name')
                if name and name.strip():
                    party_name = name.strip()

        other_ids = data.get('other_identifiers', [])
        for item in other_ids:
            if not isinstance(item, dict):
                continue
            if item.get('scheme') == 'legacy_openstates':
                identifier = item.get('identifier')
                if identifier:
                    leg_map[identifier] = party_name

    return leg_map


def build_bill_metadata_map(bills_json_path, leg_party_map):
    """
    Build a mapping from bill unique_id to (date, party).
    
    Returns dict: { unique_id: (date_string, party_abbrev) }
    """    
    bill_map = {}
    opener = gzip.open if bills_json_path.endswith('.gz') else open
    mode = "rt"

    # Pre-count lines for progress bar if not compressed
    total = None
    if opener is open:
        with open(bills_json_path, 'rt', encoding='utf-8', errors='replace') as tmp:
            total = sum(1 for _ in tmp)

    with opener(bills_json_path, mode, encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, total=total, desc="Building bill metadata", unit="lines"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            uid = obj.get('unique_id')
            if not uid:
                continue

            state = str(obj.get('state', '')).lower()
            if state in EXCLUDED_STATES:
                continue

            # Extract date
            date_created = obj.get('date_created') or ''
            date_only = ''
            if isinstance(date_created, str) and date_created:
                parts = date_created.split()[0]
                if len(parts) >= 10 and parts[0:4].isdigit():
                    date_only = parts[:10]

            # Extract party from primary sponsor
            party = 'UNK'
            sponsors = obj.get('sponsers') or obj.get('sponsors')
            if isinstance(sponsors, list):
                primary_leg_id = None
                for s in sponsors:
                    if not isinstance(s, dict):
                        continue
                    typ = s.get('type') or s.get('classification')
                    if typ and str(typ).lower() == 'primary':
                        primary_leg_id = s.get('leg_id') or s.get('legacy_openstates_id')
                        if primary_leg_id:
                            break

                if primary_leg_id:
                    party_name = leg_party_map.get(primary_leg_id)
                    if party_name:
                        abbrev = ''.join([p[0] for p in party_name.split() if p])
                        party = abbrev.upper() if abbrev else 'UNK'

            bill_map[uid] = (date_only, party)

    return bill_map


def augment_chunks_csv(chunks_csv_path, bill_map, use_party, out_csv_path):
    """Augment chunks CSV with date and user_type columns.
    
    Returns a dict with statistics about the processed data.
    """    
    # Pre-count rows for progress bar
    total_rows = None
    with open(chunks_csv_path, 'r', encoding='utf-8') as tmp:
        total_rows = max(sum(1 for _ in tmp) - 1, 0)

    dirpath = os.path.dirname(out_csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    chunk_lengths = []
    user_type_counts = Counter()
    date_counts = Counter()
    all_rows = []
    rows_written = 0

    with open(chunks_csv_path, 'r', encoding='utf-8') as infile, \
         open(out_csv_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames)

        out_fields = fieldnames + ['date', 'user_type']
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for row in tqdm(reader, total=total_rows, desc="Augmenting chunks", unit="rows"):
            uid = row.get('unique_id')
            state = (row.get('state') or '').strip().lower()

            if state in EXCLUDED_STATES:
                continue

            date_str, party = bill_map.get(uid, ('', 'UNK'))

            if use_party:
                suffix = party.upper() if party else 'UNK'
                user_type = f"{state}_{suffix}"
            else:
                user_type = state

            row['date'] = date_str
            row['user_type'] = user_type
            writer.writerow(row)
            
            rows_written += 1
            chunk_text = row.get('text', '')
            chunk_lengths.append(len(chunk_text))
            user_type_counts[user_type] += 1
            if date_str:
                date_counts[date_str[:4]] += 1
            
            if len(all_rows) < 10000:
                all_rows.append(row.copy())
            elif random.random() < 0.01:
                all_rows[random.randint(0, len(all_rows) - 1)] = row.copy()

    return {
        'rows_written': rows_written,
        'chunk_lengths': chunk_lengths,
        'user_type_counts': user_type_counts,
        'date_counts': date_counts,
        'sample_rows': all_rows
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment chunk CSV with date and user_type using raw bills and people data"
    )
    parser.add_argument(
        '--bills-json',
        default='state_bills.json',
        help='Path to raw state bills JSON lines file (optionally .gz)'
    )
    parser.add_argument(
        '--chunks-csv',
        default='preprocess_output/chunks.csv',
        help='Path to chunk CSV produced by the preprocessing pipeline'
    )
    parser.add_argument(
        '--people-dir',
        default="people/data",
        help='Path to people/data directory for party lookup'
    )
    parser.add_argument(
        '--out-csv',
        default='output/final.csv',
        help='Path to write the augmented CSV'
    )
    parser.add_argument(
        '--use-party',
        action='store_true',
        help='If set, include party in user_type (state_party)'
    )
    args = parser.parse_args()

    leg_party_map = {}
    if args.people_dir:
        leg_party_map = build_legislator_party_map(args.people_dir)
        print(f"Loaded {len(leg_party_map)} legislator party mappings")

    print("Building bill metadata map...")
    bill_map = build_bill_metadata_map(args.bills_json, leg_party_map)
    print(f"Mapped {len(bill_map)} bills to date & party")

    print("Augmenting chunks CSV...")
    stats = augment_chunks_csv(args.chunks_csv, bill_map, args.use_party, args.out_csv)
    print(f"Wrote augmented CSV to {args.out_csv}")
    
    print("\n" + "="*60)
    print("SPOT CHECK & STATISTICS")
    print("="*60)
    
    print(f"\nTotal rows written: {stats['rows_written']:,}")
    
    if stats['chunk_lengths']:
        chunk_lengths = stats['chunk_lengths']
        print(f"\nChunk length statistics:")
        print(f"   Mean: {sum(chunk_lengths) / len(chunk_lengths):.1f} chars")
        print(f"   Min: {min(chunk_lengths)} chars")
        print(f"   Max: {max(chunk_lengths)} chars")
        print(f"   Median: {sorted(chunk_lengths)[len(chunk_lengths)//2]} chars")
    
    print(f"\nUser type distribution (top 20):")
    for user_type, count in stats['user_type_counts'].most_common(20):
        pct = 100 * count / stats['rows_written']
        print(f"   {user_type:20s}: {count:6,} ({pct:5.2f}%)")
    
    total_user_types = len(stats['user_type_counts'])
    if total_user_types > 20:
        print(f"   ... and {total_user_types - 20} more user types")
    
    if stats['date_counts']:
        print(f"\nDate distribution by year (top 10):")
        for year, count in sorted(stats['date_counts'].most_common(10)):
            pct = 100 * count / stats['rows_written']
            print(f"   {year}: {count:6,} ({pct:5.2f}%)")
    
    sample_size = min(5, len(stats['sample_rows']))
    if sample_size > 0:
        print(f"\nRandom sample of {sample_size} rows:")
        sample_rows = random.sample(stats['sample_rows'], sample_size)
        for i, row in enumerate(sample_rows, 1):
            print(f"\n   --- Sample {i} ---")
            print(f"   unique_id: {row.get('unique_id', 'N/A')}")
            print(f"   state: {row.get('state', 'N/A')}")
            print(f"   user_type: {row.get('user_type', 'N/A')}")
            print(f"   date: {row.get('date', 'N/A')}")
            text = row.get('text', '')
            print(f"   text (first 100 chars): {text[:100]}...")
            print(f"   text length: {len(text)} chars")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
