# FinanceBench Evaluation

Benchmarking [FinanceBench](https://github.com/patronus-ai/financebench) against the PromptPulse API — comparing a plain Gemini model vs Gemini with the Elen AI Super Agent.

## What This Does

For each of the 150 questions in FinanceBench, the script runs four steps in a single pass:

1. **Plain model** — creates a conversation, uploads the source PDF, sends the question with evidence text, records the answer
2. **Elen model** — does the same with the Elen agent attached
3. **LLM judge** — sends both answers to the same model and asks whether each is equivalent to the gold answer
4. **Save** — writes both output files so progress is never lost if the script is interrupted

Context is provided two ways at once: the relevant evidence text from the dataset is included in the message body, and the full PDF is uploaded as an attached file.

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

Run on a small sample first to verify everything works:
```bash
uv run python benchmark.py --limit 5
```

Run the full benchmark:
```bash
uv run python benchmark.py
```

If the script is interrupted, just run it again — it will resume from where it stopped.

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
| `gold_answer` | Correct human-annotated answer |
| `Plain model` | Answer from plain model |
| `plain_is_correct` | Judge verdict for plain model (True/False) |
| `plain_judge_response` | Raw judge reasoning for manual review |
| `Model + Elen` | Answer from model with Elen agent |
| `elen_is_correct` | Judge verdict for Elen model (True/False) |
| `elen_judge_response` | Raw judge reasoning for manual review |

At the end of the run, a score summary is printed:
```
==================================================
RESULTS
==================================================
Questions evaluated : 150
Plain model score   : 68/150 (45.3%)
Elen  model score   : 95/150 (63.3%)
```

## How the LLM Judge Works

After getting answers from both models, the script creates a temporary conversation and asks the same model to evaluate whether the answer is equivalent to the gold answer. Numbers are considered equivalent if they differ by less than 5%. The judge's raw response is saved to the comparison table so you can review any cases where the verdict seems wrong.

## Rate Limits

Gemini enforces RPM (requests per minute) and TPM (tokens per minute) limits. The script waits 5 seconds between model calls and 1 second between judge calls. If you hit rate limit errors, increase `SLEEP_BETWEEN_REQUESTS` in `benchmark.py`.

## How the API Works

The PromptPulse API uses a conversation-based model:

1. **POST** `/conversations` — create a conversation, optionally attaching a model and an AI Super Agent
2. **POST** `/conversations/{id}/files` — upload a file (PDF), get back a `file_id`
3. **POST** `/conversations/{id}/messages` — send a message with optional `fileIds`, get back the assistant's response
4. **DELETE** `/conversations/{id}` — clean up after each question

Authentication uses a custom header: `x-api-token: YOUR_KEY`