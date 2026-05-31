# FinanceBench Evaluation

Benchmarking [FinanceBench](https://github.com/patronus-ai/financebench) against the PromptPulse API — comparing a plain Gemini model vs Gemini with the Elen AI Super Agent.

## What This Does

The benchmark runs in two separate passes:

**Pass 1 — Collection** (`python benchmark.py`)
For each question: creates a conversation, uploads the source PDF, sends the question with evidence text, records the answer for both plain model and Elen. Saves to `raw_results.csv` after every question.

**Pass 2 — Judgment** (`python benchmark.py --judge`)
Reads `raw_results.csv`, sends each answer pair to an LLM judge, saves verdicts to `comparison_table.csv` after every question.

Both passes support resume — if interrupted, just run again and they pick up where they stopped.

## Project Structure

```
financebench-eval/
├── benchmark.py         — main script
├── .env                 — API credentials (never committed to git)
├── .env.example         — template showing required variables
├── data/
│   └── financebench_open_source.jsonl  — 150 questions with answers and evidence
├── pdfs/                — source PDF documents (not committed to git)
└── results/
    ├── raw_results.csv       — full raw answers from both models
    └── comparison_table.csv  — side-by-side comparison with judge verdicts
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/financebench-eval.git
cd financebench-eval
```

**2. Install dependencies**
```bash
uv init
uv add pandas requests tqdm python-dotenv
```

**3. Create your `.env` file**
```bash
cp .env.example .env
```
Then fill in your values:
```
API_KEY=your_api_key_here
BASE_URL=https://llm-manager.etacar.io/api/external
MODEL_ID=google:gemini-2.5-flash
ELEN_AGENT_ID=your_elen_agent_id_here
```

To find available model IDs:
```bash
curl -X GET "$BASE_URL/models" -H "x-api-token: $API_KEY"
```

To find the Elen agent ID:
```bash
curl -X GET "$BASE_URL/ai-super-agents" -H "x-api-token: $API_KEY"
```

**4. Add the data files**

Copy the dataset and PDFs from a local FinanceBench clone:
```bash
mkdir data results
cp /path/to/financebench/data/financebench_open_source.jsonl data/
cp -r /path/to/financebench/pdfs pdfs/
```

## Usage

**Step 1 — collect answers** (test on 5 questions first):
```bash
uv run python benchmark.py --limit 5
uv run python benchmark.py          # full 150 questions
```

**Step 2 — run judgment:**
```bash
uv run python benchmark.py --judge
```

**Re-judge everything from scratch:**
```bash
uv run python benchmark.py --judge --force-rejudge
```

## Output

**`results/raw_results.csv`** — full raw answers from both models:

| Column | Description |
|---|---|
| `financebench_id` | Question ID from the dataset |
| `company` | Company the question is about |
| `doc_name` | Source PDF filename |
| `question` | The question asked |
| `gold_answer` | Correct human-annotated answer |
| `plain_answer` | Full answer from plain model |
| `elen_answer` | Full answer from model with Elen agent |

**`results/comparison_table.csv`** — side-by-side comparison with judge verdicts:

| Column | Description |
|---|---|
| `financebench_id` | Question ID (used for resume) |
| `gold_answer` | Correct human-annotated answer |
| `Plain model` | Answer from plain model |
| `plain_is_correct` | Judge verdict for plain model (True/False) |
| `plain_judge_response` | Raw judge reasoning for manual review |
| `Model + Elen` | Answer from model with Elen agent |
| `elen_is_correct` | Judge verdict for Elen model (True/False) |
| `elen_judge_response` | Raw judge reasoning for manual review |

At the end of the judgment pass, a score summary is printed:
```
==================================================
RESULTS
==================================================
Questions evaluated : 150
Plain model score   : 68/150 (45.3%)
Elen  model score   : 95/150 (63.3%)
```

## How the LLM Judge Works

The judge creates a temporary conversation and asks the model whether the answer is equivalent to the gold answer. Numbers differing by less than 5% are considered equivalent. YES/NO is extracted using regex so punctuation like `YES.` or `YES,` is handled correctly. The raw judge response is saved to the comparison table for manual review of any questionable verdicts.

## Rate Limits

Gemini enforces RPM (requests per minute) and TPM (tokens per minute) limits. The script waits 5 seconds between model calls and 1 second between judge calls. If you hit rate limit errors, increase `SLEEP_BETWEEN_REQUESTS` in `benchmark.py`.

## How the API Works

The PromptPulse API uses a conversation-based model:

1. **POST** `/conversations` — create a conversation, optionally attaching a model and an AI Super Agent
2. **POST** `/conversations/{id}/files` — upload a file (PDF), get back a `file_id`
3. **POST** `/conversations/{id}/messages` — send a message with optional `fileIds`, get back the assistant's response
4. **DELETE** `/conversations/{id}` — clean up after each question

Authentication uses a custom header: `x-api-token: YOUR_KEY`