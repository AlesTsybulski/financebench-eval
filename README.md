# FinanceBench Evaluation

Benchmarking [FinanceBench](https://github.com/patronus-ai/financebench) against the PromptPulse API — comparing a plain Gemini model vs Gemini with the Elen AI Super Agent.

## What This Does

For each of the 150 questions in FinanceBench, the script:

1. Creates a conversation with the plain model, uploads the source PDF, sends the question with evidence text, and records the answer
2. Does the same thing with the Elen agent attached to the conversation
3. Saves both answers to two CSV files after every question

Context is provided two ways at once: the relevant evidence text extracted from the document is included in the message body, and the full PDF is uploaded as an attached file. This gives the model both a focused excerpt and access to the complete source document.

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
    └── comparison_table.csv  — plain model vs Elen, final answers only
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

Two files are saved to `results/`:

**`raw_results.csv`** — full answers from both models:

| Column | Description |
|---|---|
| `financebench_id` | Question ID from the dataset |
| `company` | Company the question is about |
| `doc_name` | Source PDF filename |
| `question` | The question asked |
| `gold_answer` | Correct human-annotated answer |
| `plain_answer` | Full answer from plain model |
| `elen_answer` | Full answer from model with Elen agent |

**`comparison_table.csv`** — clean side-by-side comparison:

| Column | Description |
|---|---|
| `Plain model` | Final answer from plain model |
| `Model + Elen` | Final answer from model with Elen agent |

## Rate Limits

Gemini enforces RPM (requests per minute) and TPM (tokens per minute) limits. The script waits 5 seconds between each question to stay within these limits. If you hit rate limit errors, increase `SLEEP_BETWEEN_REQUESTS` in `benchmark.py`.

## How the API Works

The PromptPulse API uses a conversation-based model:

1. **POST** `/conversations` — create a conversation, optionally attaching a model and an AI Super Agent
2. **POST** `/conversations/{id}/files` — upload a file (PDF), get back a `file_id`
3. **POST** `/conversations/{id}/messages` — send a message with optional `fileIds`, get back the assistant's response
4. **DELETE** `/conversations/{id}` — clean up after each question

Authentication uses a custom header: `x-api-token: YOUR_KEY`