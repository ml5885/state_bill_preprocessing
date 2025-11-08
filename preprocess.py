"""
Main preprocessing pipeline for state legislative bills.

Cleans boilerplate text and splits bills into paragraph-sized chunks.
"""
import argparse
import gc
from pathlib import Path

import pandas as pd

import boilerplate
import chunking


def main():
    parser = argparse.ArgumentParser(
        description="Clean and chunk state bill documents for agenda setting"
    )
    parser.add_argument(
        "--input",
        default="state_bills.json",
        help="Path to the raw NDJSON file (optionally .gz) containing state bills",
    )
    parser.add_argument(
        "--outdir",
        default="preprocess_output",
        help="Directory where cleaned and chunked CSVs will be written",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Load existing outputs from outdir and print summary statistics without reprocessing",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Only process a single state (case-insensitive). Example: --state texas or --state ca",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cleaned_csv_path = outdir / "cleaned_bills.csv"
    chunks_csv_path = outdir / "chunks.csv"

    if args.analysis:
        if not cleaned_csv_path.exists():
            raise SystemExit(f"Missing cleaned CSV for analysis: {cleaned_csv_path}")
        if not chunks_csv_path.exists():
            raise SystemExit(f"Missing chunks CSV for analysis: {chunks_csv_path}")

        df_clean = pd.read_csv(cleaned_csv_path, dtype={'session': str})
        df_chunks = pd.read_csv(chunks_csv_path, dtype={'session': str})

        if args.state:
            state = args.state.lower()
            df_clean = df_clean[df_clean["state"].str.lower() == state]
            df_chunks = df_chunks[df_chunks["state"].str.lower() == state]

        # Derive per-state stats from cleaned data
        stats = {}
        for state, grp in df_clean.groupby('state'):
            docs = len(grp)
            bills = grp[['state', 'bill_id', 'bill_version']].drop_duplicates().shape[0]
            wb = grp['words_before'].sum()
            wa = grp['words_after'].sum()
            stats[state] = {
                'bills_total': bills,
                'bills_with_text': bills,
                'documents': docs,
                'words_before': int(wb),
                'words_after': int(wa),
            }

        print("\n=== Boilerplate Removal Stats ===")
        boilerplate.print_summary_table(stats)
        print("\n=== Chunking Stats ===")
        chunking.print_chunk_stats(df_chunks)
        return

    # Step 1: Remove boilerplate and write cleaned CSV
    print("[Step 1] Removing boilerplate...")
    states = [args.state.lower()] if args.state else None
    
    df_clean, stats = boilerplate.process_bills(args.input, states=states)
    
    df_clean.to_csv(cleaned_csv_path, index=False)
    print(f"Wrote cleaned CSV to {cleaned_csv_path}")
    boilerplate.print_summary_table(stats)

    del df_clean
    del stats
    gc.collect()

    # Step 2: Chunk cleaned documents (streamed to avoid memory issues)
    print("\n[Step 2] Chunking cleaned documents...")
    
    spot_check_dir = outdir / "spot_check"
    chunk_stats = chunking.chunk_csv(
        str(cleaned_csv_path), 
        str(chunks_csv_path), 
        chunksize=1000,
        spot_check_dir=str(spot_check_dir)
    )
    
    print(f"Wrote chunked CSV to {chunks_csv_path}")
    chunking.print_chunk_stats_summary(chunk_stats)
    
    print(f"\nSpot check files saved to {spot_check_dir}/")
    print("Review these files to verify chunking quality.")


if __name__ == "__main__":
    main()
