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

Run the pipeline (this takes roughly 2 hours on a Macbook Pro with 16GB RAM):

```bash
# Step 1: Remove boilerplate and chunk bills
python preprocess.py --input state_bills.json

# Step 2: Add metadata (date, party)
python create_final_df.py
```

## Pipeline Steps

### 1. Boilerplate Removal (`boilerplate.py`)

Removes boilerplate using a three-stage approach:

1. **Sentence-based frequency**: Sentences appearing in >=5% of documents (minimum 5 documents) within each state are flagged as boilerplate.
2. **Section-title removal**: Removes entire sections with common boilerplate titles (e.g., "Effective Date", "Severability").
3. **Tiny-chunk frequency**: Aggressively removes small chunks (100 chars) appearing in >=1% of documents.

Also removes line numbers, XML/HTML/PDF metadata, and state-specific patterns (e.g., Massachusetts petition headers).

### 2. Chunking (`chunking.py`)

Splits cleaned bills into paragraph-sized chunks using:

- Section headings (Section 1, Sec. 2, etc.)
- Runs of 2+ newlines
- Subsection markers like (a), (b), (1)
- Line-level heuristics for documents with minimal structure

Chunks are constrained to 250-750 characters: short chunks are merged with neighbors, long chunks are split at subsection boundaries.

### 3. Metadata Augmentation (`create_final_df.py`)

Adds two columns to each chunk:

- `date`: bill creation date (YYYY-MM-DD)
- `user_type`: state_party identifier (e.g., 'ca_D', 'ca_R') - always includes party

Requires the raw bills JSON and the OpenStates people directory for party affiliation. Outputs a pickle file with `post_text` column (renamed from `bill_text`).

## Files

- `preprocess.py` - main entry point for cleaning and chunking
- `boilerplate.py` - boilerplate detection and removal
- `chunking.py` - bill text splitting logic
- `create_final_df.py` - metadata augmentation
- `requirements.txt` - Python dependencies

## Output Format

The final pickle file (`output/cleaned_final_df.pkl`) contains these columns:

- `state` - two-letter state code
- `session` - legislative session identifier
- `bill_id` - legislature's bill identifier
- `unique_id` - unique identifier for this bill
- `sunlight_id` - Sunlight Foundation identifier
- `bill_version` - 'first' or 'last'
- `chunk_id` - chunk identifier within document (e.g., 'chunk_1')
- `post_text` - cleaned chunk text (renamed from `bill_text`)
- `date` - bill creation date (YYYY-MM-DD)
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
- `--out-csv` - temporary CSV output path (default: `output/final.csv`, deleted after conversion)
- `--out-pkl` - final pickle output path (default: `output/cleaned_final_df.pkl`)

## Data

Download the state bill data (state_bills.json.gz) from Harvard Dataverse:

- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FCZ25GF

Note: The compressed file is large (~2.5GB). The uncompressed JSON file is larger (~15GB).

After downloading:

```bash
gunzip -k state_bills.json.gz
```

Use the resulting state_bills.json with the commands above.

Citation: Replication Data for: Measuring Policy Similarity Through Bill Text Reuse. Original data from openstates.org.

### OpenStates people directory (party metadata)

Download the OpenStates people repository and point --people-dir to its data folder:

```bash
# Clone the repository inside the current directory
git clone https://github.com/openstates/people

# Then run create_final_df.py with:
python create_final_df.py \
  --bills-json state_bills.json \
  --chunks-csv preprocess_output/chunks.csv \
  --people-dir people/data
```
