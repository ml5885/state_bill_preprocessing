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

import pandas as pd
import yaml
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

EXCLUDED_STATES = {'co', 'pr', 'dc'}


def build_legislator_party_map(people_dir):
    """
    Build mappings from legacy OpenStates legislator IDs to party names,
    and from (state, name) to (leg_id, party) for fallback lookups.
    
    Returns tuple: (leg_map, name_map)
      - leg_map: { legislator_id: party_name }
      - name_map: { (state, normalized_name): {'leg_id': ..., 'party': ...} }
    """
    leg_map = {}
    name_map = {}
    if not os.path.isdir(people_dir):
        print(f"[WARN] people_dir not found: {people_dir}")
        return leg_map, name_map

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

        # Get state from roles  
        state = None
        roles = data.get('roles', [])
        if isinstance(roles, list) and roles:
            for role in roles:
                if isinstance(role, dict):
                    jurisdiction = role.get('jurisdiction', '')
                    # Extract state from OCD-ID like: ocd-jurisdiction/country:us/state:ak/government
                    if 'state:' in jurisdiction:
                        parts = jurisdiction.split('state:')
                        if len(parts) > 1:
                            state = parts[1].split('/')[0].split(':')[0]
                            break
        
        # Get legacy leg_id
        leg_id = None
        other_ids = data.get('other_identifiers', [])
        for item in other_ids:
            if not isinstance(item, dict):
                continue
            if item.get('scheme') == 'legacy_openstates':
                identifier = item.get('identifier')
                if identifier:
                    leg_map[identifier] = party_name
                    leg_id = identifier
                    break
        
        # Build name map for fallback lookups
        if state and leg_id and party_name != 'Unknown':
            person_name = data.get('name', '')
            if person_name:
                # Normalize: lowercase, remove extra spaces
                norm_name = ' '.join(person_name.lower().split())
                key = (state.lower(), norm_name)
                name_map[key] = {'leg_id': leg_id, 'party': party_name}

    return leg_map, name_map


def normalize_party(party_name, party_mappings=None):
    """
    Normalize a party name to standard abbreviations.
    Returns 'D' for Democratic, 'R' for Republican, or first letter(s) for others.
    If party_mappings dict is provided, records the mapping.
    """
    if not party_name or party_name == 'Unknown':
        return None
    
    party_lower = party_name.lower()
    
    # Check for Democratic
    if 'democrat' in party_lower:
        abbrev = 'D'
    # Check for Republican
    elif 'republican' in party_lower:
        abbrev = 'R'
    elif 'green' in party_lower or 'progressive' in party_lower:
        abbrev = 'D'
    elif 'forward' in party_lower:
        abbrev = 'I'
    else:
        abbrev = ''.join([p[0] for p in party_name.split() if p])
        abbrev = abbrev.upper() if abbrev else None
    
    # Track the mapping
    if abbrev and party_mappings is not None:
        if abbrev not in party_mappings:
            party_mappings[abbrev] = set()
        party_mappings[abbrev].add(party_name)
    
    return abbrev


def normalize_name(name):
    """Normalize a name for matching: lowercase, remove extra spaces."""
    if not name:
        return ''
    return ' '.join(str(name).lower().split())


def lookup_party_by_name(state, name, name_map):
    """Try to find a legislator's party by name lookup."""
    if not name or not state:
        return None
    norm_name = normalize_name(name)
    key = (state.lower(), norm_name)
    result = name_map.get(key)
    if result:
        return result['party']
    return None


def get_majority_party(sponsors, leg_party_map, name_map, state, party_mappings=None):
    """
    Get the majority party from a list of sponsors.
    Returns (party, reason) tuple.
    """
    party_counts = Counter()
    
    for s in sponsors:
        if not isinstance(s, dict):
            continue
        
        # Try leg_id first
        leg_id = s.get('leg_id') or s.get('legacy_openstates_id')
        party_name = None
        
        if leg_id and leg_id in leg_party_map:
            party_name = leg_party_map[leg_id]
        elif not leg_id:
            # Try name lookup
            name = s.get('name')
            if name:
                party_name = lookup_party_by_name(state, name, name_map)
        
        if party_name:
            party_abbrev = normalize_party(party_name, party_mappings)
            if party_abbrev:
                party_counts[party_abbrev] += 1
    
    if not party_counts:
        return None, 'no_parties_found'
    
    # Get majority party, tie-break by first occurrence
    most_common = party_counts.most_common(1)[0][0]
    return most_common, 'majority_party'


def build_bill_metadata_map(bills_json_path, leg_party_map, name_map):
    """
    Build a mapping from bill unique_id to (date, party).
    
    Returns dict: { unique_id: (date_string, party_abbrev) }
    """    
    bill_map = {}
    opener = gzip.open if bills_json_path.endswith('.gz') else open
    mode = "rt"

    # Track by state
    state_counters = {}
    
    # Track party mappings: abbrev -> set of original party names
    party_mappings = {}

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
            
            # Initialize state counter
            if state not in state_counters:
                state_counters[state] = {
                    'total': 0,
                    'no_sponsors': 0,
                    'no_primary_sponsor': 0,
                    'primary_leg_id_missing': 0,
                    'leg_id_not_in_map': 0,
                    'party_found': 0,
                }
            state_counters[state]['total'] += 1

            # Extract date
            date_created = obj.get('date_created') or ''
            date_only = ''
            if isinstance(date_created, str) and date_created:
                parts = date_created.split()[0]
                if len(parts) >= 10 and parts[0:4].isdigit():
                    date_only = parts[:10]

            # Extract party from primary sponsor with fallback strategy
            party = None
            party_source = None
            sponsors = obj.get('sponsers') or obj.get('sponsors')
            
            if not isinstance(sponsors, list) or not sponsors:
                state_counters[state]['no_sponsors'] += 1
                # Skip bills with no sponsors - we'll filter these out
                continue
            
            # Strategy 1: Try to find primary sponsor
            primary_sponsor = None
            primary_name = None
            for s in sponsors:
                if not isinstance(s, dict):
                    continue
                typ = s.get('type') or s.get('classification')
                if typ and str(typ).lower() == 'primary':
                    primary_sponsor = s
                    primary_name = s.get('name')
                    break
            
            if primary_sponsor:
                # Have primary sponsor - try leg_id first
                leg_id = primary_sponsor.get('leg_id') or primary_sponsor.get('legacy_openstates_id')
                
                if leg_id:
                    # Strategy 1a: Look up by leg_id
                    party_name = leg_party_map.get(leg_id)
                    if party_name:
                        party = normalize_party(party_name, party_mappings)
                        party_source = 'primary_leg_id'
                
                if not party and primary_name:
                    # Strategy 1b: Look up by name
                    party_name = lookup_party_by_name(state, primary_name, name_map)
                    if party_name:
                        party = normalize_party(party_name, party_mappings)
                        party_source = 'primary_name_lookup'
            
            # Strategy 2: No primary or primary failed - try cosponsors
            if not party:
                cosponsors = [s for s in sponsors if isinstance(s, dict)]
                if cosponsors:
                    if len(cosponsors) == 1:
                        # Single cosponsor - use it
                        s = cosponsors[0]
                        leg_id = s.get('leg_id') or s.get('legacy_openstates_id')
                        if leg_id:
                            party_name = leg_party_map.get(leg_id)
                            if party_name:
                                party = normalize_party(party_name, party_mappings)
                                party_source = 'single_cosponsor'
                        
                        if not party:
                            # Try name lookup for single cosponsor
                            name = s.get('name')
                            if name:
                                party_name = lookup_party_by_name(state, name, name_map)
                                if party_name:
                                    party = normalize_party(party_name, party_mappings)
                                    party_source = 'single_cosponsor_name'
                    else:
                        # Multiple cosponsors - get majority party
                        party, reason = get_majority_party(cosponsors, leg_party_map, name_map, state, party_mappings)
                        if party:
                            party_source = 'cosponsor_majority'
            
            # Final decision: keep or skip this bill
            if party:
                state_counters[state]['party_found'] += 1
                bill_map[uid] = (date_only, party)
            else:
                # Track why we're skipping this bill
                if not primary_sponsor:
                    state_counters[state]['no_primary_sponsor'] += 1
                elif primary_sponsor and not (primary_sponsor.get('leg_id') or primary_sponsor.get('legacy_openstates_id')):
                    state_counters[state]['primary_leg_id_missing'] += 1
                else:
                    state_counters[state]['leg_id_not_in_map'] += 1
                
                # Skip this bill - don't add to bill_map
                continue

    # Print party mappings
    print("\n" + "="*60)
    print("PARTY ABBREVIATION MAPPINGS")
    print("="*60)
    for abbrev in sorted(party_mappings.keys()):
        original_names = sorted(party_mappings[abbrev])
        if len(original_names) == 1:
            print(f"  {abbrev} = {original_names[0]}")
        else:
            print(f"  {abbrev} = {', '.join(original_names)}")
    print("="*60 + "\n")

    return bill_map


def augment_chunks_csv(chunks_csv_path, bill_map, out_csv_path):
    """Augment chunks CSV with date and user_type columns.
    
    user_type will always include party (state_party format).
    Returns a dict with statistics about the processed data.
    """    
    # Pre-count rows for progress bar
    total_rows = None
    with open(chunks_csv_path, 'r', encoding='utf-8') as tmp:
        total_rows = max(sum(1 for _ in tmp) - 1, 0)

    dirpath = os.path.dirname(out_csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    user_type_counts = Counter()
    date_counts = Counter()
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

            # Skip rows where we don't have metadata (bill was excluded during metadata building)
            if uid not in bill_map:
                continue
            
            date_str, party = bill_map[uid]
            
            # Double-check: skip any bills with UNK party (shouldn't happen, but safety check)
            if not party or party == 'UNK':
                continue

            # Always use party in user_type
            suffix = party.upper() if party else 'UNK'
            user_type = f"{state}_{suffix}"

            row['date'] = date_str
            row['user_type'] = user_type
            writer.writerow(row)
            
            rows_written += 1
            user_type_counts[user_type] += 1
            if date_str:
                date_counts[date_str[:4]] += 1

    return {
        'rows_written': rows_written,
        'user_type_counts': user_type_counts,
        'date_counts': date_counts,
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
        help='Path to write the augmented CSV (temporary, will be converted to pickle)'
    )
    parser.add_argument(
        '--out-pkl',
        default='output/cleaned_final_df.pkl',
        help='Path to write the final pickle file'
    )
    args = parser.parse_args()

    leg_party_map = {}
    name_map = {}
    if args.people_dir:
        leg_party_map, name_map = build_legislator_party_map(args.people_dir)
        print(f"Loaded {len(leg_party_map)} legislator party mappings")
        print(f"Loaded {len(name_map)} name-based party mappings")

    print("Building bill metadata map...")
    bill_map = build_bill_metadata_map(args.bills_json, leg_party_map, name_map)
    print(f"Mapped {len(bill_map)} bills to date & party")

    print("Augmenting chunks CSV (adding date and user_type columns)...")
    stats = augment_chunks_csv(args.chunks_csv, bill_map, args.out_csv)
    print(f"Wrote temporary CSV to {args.out_csv}")
    
    print("Converting to DataFrame and saving as pickle...")
    df = pd.read_csv(args.out_csv)
    df.rename(columns={'bill_text': 'post_text'}, inplace=True)
    
    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    df.to_pickle(args.out_pkl)
    print(f"Saved final DataFrame to {args.out_pkl}")
    
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)
        print(f"Removed temporary CSV file")
    
    print("\n" + "="*60)
    print("FINAL OUTPUT STATISTICS")
    print("="*60)
    
    print(f"\nTotal chunks written: {stats['rows_written']:,}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    print(f"\nUser type distribution:")
    for user_type, count in sorted(stats['user_type_counts'].items()):
        pct = 100 * count / stats['rows_written']
        print(f"   {user_type:19s}: {count:8,} ({pct:5.2f}%)")
    
    if stats['date_counts']:
        print(f"\nDate distribution by year:")
        for year, count in sorted(stats['date_counts'].items()):
            pct = 100 * count / stats['rows_written']
            print(f"   {year}: {count:8,} ({pct:5.2f}%)")
    
    try:
        print("\nExample rows:")
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        n_examples = min(5, len(df))
        if n_examples > 0:
            sample = df.sample(n=n_examples, random_state=42)
            for _, row in sample.iterrows():
                print("\n" + "-"*40)
                print(row.to_string())
        else:
            print("  (no rows in DataFrame)")
    finally:
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.max_columns')
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
