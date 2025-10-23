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
from pathlib import Path

import yaml
from tqdm import tqdm


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
    """Augment chunks CSV with date and user_type columns."""
    # Pre-count rows for progress bar
    total_rows = None
    with open(chunks_csv_path, 'r', encoding='utf-8') as tmp:
        total_rows = max(sum(1 for _ in tmp) - 1, 0)

    dirpath = os.path.dirname(out_csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

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

            date_str, party = bill_map.get(uid, ('', 'UNK'))

            if use_party:
                suffix = party.upper() if party else 'UNK'
                user_type = f"{state}_{suffix}"
            else:
                user_type = state

            row['date'] = date_str
            row['user_type'] = user_type
            writer.writerow(row)


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
    augment_chunks_csv(args.chunks_csv, bill_map, args.use_party, args.out_csv)
    print(f"Wrote augmented CSV to {args.out_csv}")


if __name__ == '__main__':
    main()
