# State Bill Preprocessing

Clean and chunk U.S. state legislative bill texts for text analysis.

The pipeline removes boilerplate text and splits bills into paragraph-sized chunks suitable for embedding models and text reuse detection.

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline:

```bash
# Step 1: Remove boilerplate and chunk bills
python preprocess.py --input state_bills.json --outdir output

# Step 2: Add metadata (date, party)
python create_final_df.py \
    --bills-json state_bills.json \
    --chunks-csv output/chunks.csv \
    --people-dir people/data \
    --out-csv output/final.csv \
    --use-party
```

## Pipeline Steps

### 1. Boilerplate Removal (`boilerplate.py`)

Identifies and removes common boilerplate sentences that appear across many bills within each state. Uses document frequency: sentences appearing in >10% of documents (minimum 5 documents) are flagged as boilerplate.

Also removes line numbers that appear at the start or end of lines.

### 2. Chunking (`chunking.py`)

Splits cleaned bills into paragraph-sized chunks using:

- Section headings (Section 1, Sec. 2, etc.)
- Runs of 2+ newlines
- Line-level heuristics for documents with minimal structure

Chunks shorter than 250 characters are merged with neighboring chunks to ensure meaningful units.

### 3. Metadata Augmentation (`create_final_df.py`)

Adds two columns to each chunk, which are needed for downstream analysis (e.g. agenda setting):

- `date`: bill creation date (YYYY-MM-DD)
- `user_type`: state code (e.g., 'ca') or state_party (e.g., 'ca_D') if `--use-party` is set

Requires the raw bills JSON and optionally the OpenStates people directory for party affiliation.

## Files

- `preprocess.py` - main entry point for cleaning and chunking
- `boilerplate.py` - boilerplate detection and removal
- `chunking.py` - bill text splitting logic
- `create_final_df.py` - metadata augmentation
- `requirements.txt` - Python dependencies

## Output Format

The final CSV contains these columns:

- `state` - two-letter state code
- `session` - legislative session identifier
- `bill_id` - legislature's bill identifier
- `unique_id` - unique identifier for this bill
- `sunlight_id` - Sunlight Foundation identifier
- `bill_version` - 'first' or 'last'
- `chunk_id` - chunk identifier within document (e.g., 'chunk_1')
- `bill_text` - cleaned chunk text
- `date` - bill creation date
- `user_type` - state or state_party identifier

## Command-Line Options

### `preprocess.py`

- `--input` - path to raw bills NDJSON file (default: `state_bills.json`)
- `--outdir` - output directory (default: `preprocess_output`)
- `--state` - process only one state (e.g., `--state ca`)
- `--analysis` - print statistics from existing output without reprocessing

### `create_final_df.py`

- `--bills-json` - path to raw bills JSON (default: `state_bills.json`)
- `--chunks-csv` - path to chunks CSV (default: `preprocess_output/chunks.csv`)
- `--people-dir` - path to OpenStates people data (default: `people/data`)
- `--out-csv` - output path (default: `output/final.csv`)
- `--use-party` - include party in user_type

## Data

Download the state bill data (state_bills.json.gz) from Harvard Dataverse:

- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FCZ25GF

Note: The compressed file is large (~2.5GB). The uncompressed CSV is larger (~15GB).

After downloading:

```bash
gunzip -k state_bills.json.gz
```

Use the resulting state_bills.json with the commands above.

Citation: Replication Data for: Measuring Policy Similarity Through Bill Text Reuse. Original data from openstates.org.

### OpenStates people directory (party metadata)

If using --use-party, download the OpenStates people repository and point --people-dir to its data folder:

```bash
# Clone the repository inside the current directory
git clone https://github.com/openstates/people

# Then run create_final_df.py with:
python create_final_df.py \
  --bills-json state_bills.json \
  --chunks-csv output/chunks.csv \
  --people-dir people/data \
  --out-csv output/final.csv \
  --use-party
```
