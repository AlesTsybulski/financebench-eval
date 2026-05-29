import os
import re
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY       = os.getenv("API_KEY")
BASE_URL      = os.getenv("BASE_URL")
MODEL_ID      = os.getenv("MODEL_ID")
ELEN_AGENT_ID = os.getenv("ELEN_AGENT_ID")

HEADERS = {
    "x-api-token": API_KEY,
    "Content-Type": "application/json",
}

SLEEP_BETWEEN_REQUESTS = 5
RAW_RESULTS_PATH = "results/raw_results.csv"


def create_conversation(title: str, agent_id: str = None) -> str | None:
    body = {"title": title, "model": MODEL_ID}
    if agent_id:
        body["aiSuperAgent"] = agent_id
    try:
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=HEADERS,
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["_id"]
    except Exception as e:
        print(f"  [ERROR] create_conversation failed: {e}")
        return None


def upload_pdf(conversation_id: str, pdf_path: str) -> str | None:
    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/conversations/{conversation_id}/files",
                headers={"x-api-token": API_KEY},
                files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
                timeout=60,
            )
        response.raise_for_status()
        files = response.json().get("files", [])
        if files:
            return files[0]["_id"]
        return None
    except Exception as e:
        print(f"  [ERROR] upload_pdf failed: {e}")
        return None


def send_message(conversation_id: str, question: str, evidence: str = None, file_id: str = None) -> str | None:
    if evidence:
        message_text = f"Context from financial document:\n{evidence}\n\nQuestion: {question}"
    else:
        message_text = question

    body = {"message": message_text}
    if file_id:
        body["fileIds"] = [file_id]

    try:
        response = requests.post(
            f"{BASE_URL}/conversations/{conversation_id}/messages",
            headers=HEADERS,
            json=body,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["assistantMessage"]["content"]
    except Exception as e:
        print(f"  [ERROR] send_message failed: {e}")
        return None


def delete_conversation(conversation_id: str):
    try:
        requests.delete(
            f"{BASE_URL}/conversations/{conversation_id}",
            headers=HEADERS,
            timeout=15,
        )
    except Exception:
        pass


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
    if not paragraphs:
        return ""
    return paragraphs[-1]


def load_checkpoint() -> tuple[list, set]:
    if os.path.exists(RAW_RESULTS_PATH) and os.path.getsize(RAW_RESULTS_PATH) > 0:
        df = pd.read_csv(RAW_RESULTS_PATH)
        done_ids = set(df["financebench_id"].astype(str).tolist())
        results  = df.to_dict("records")
        print(f"Checkpoint found — {len(done_ids)} questions already done, resuming...")
        return results, done_ids
    return [], set()


def run_benchmark(limit: int = None):
    data_path = "data/financebench_open_source.jsonl"
    rows = []
    with open(data_path, "r") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    if limit:
        rows = rows[:limit]

    results, done_ids = load_checkpoint()

    remaining = [r for r in rows if str(r["financebench_id"]) not in done_ids]

    print(f"Total questions : {len(rows)}")
    print(f"Already done    : {len(done_ids)}")
    print(f"Remaining       : {len(remaining)}")
    print(f"Model           : {MODEL_ID}")
    print(f"Elen agent      : {ELEN_AGENT_ID}\n")

    if not remaining:
        print("All questions already processed!")
    
    for i, row in enumerate(tqdm(remaining, desc="Questions")):
        question_id   = row["financebench_id"]
        question      = row["question"]
        gold_answer   = row["answer"]
        doc_name      = row["doc_name"]
        company       = row.get("company", "")

        evidence_text = None
        if row.get("evidence"):
            evidence_text = row["evidence"][0].get("evidence_text", None)

        pdf_path   = f"pdfs/{doc_name}.pdf"
        pdf_exists = os.path.exists(pdf_path)
        if not pdf_exists:
            print(f"  [WARN] PDF not found: {pdf_path} — using evidence text only")

        print(f"\n[{i+1}/{len(remaining)}] {company} | {question[:60]}...")

        plain_answer = None
        conv_id = create_conversation(f"plain_{question_id}")
        if conv_id:
            file_id = upload_pdf(conv_id, pdf_path) if pdf_exists else None
            plain_answer = send_message(conv_id, question, evidence=evidence_text, file_id=file_id)
            delete_conversation(conv_id)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        elen_answer = None
        conv_id = create_conversation(f"elen_{question_id}", agent_id=ELEN_AGENT_ID)
        if conv_id:
            file_id = upload_pdf(conv_id, pdf_path) if pdf_exists else None
            elen_answer = send_message(conv_id, question, evidence=evidence_text, file_id=file_id)
            delete_conversation(conv_id)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        results.append({
            "financebench_id": question_id,
            "company":         company,
            "doc_name":        doc_name,
            "question":        question,
            "gold_answer":     gold_answer,
            "plain_answer":    plain_answer,
            "elen_answer":     elen_answer,
        })

        os.makedirs("results", exist_ok=True)
        pd.DataFrame(results).to_csv(RAW_RESULTS_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(RAW_RESULTS_PATH, index=False)

    comparison = pd.DataFrame({
        "Plain model":  df["plain_answer"].apply(extract_final_answer),
        "Model + Elen": df["elen_answer"].apply(extract_final_answer),
    })
    comparison.to_csv("results/comparison_table.csv", index=False)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Questions evaluated: {len(df)}")
    print(f"\nFiles saved:")
    print(f"  results/raw_results.csv       — full raw answers")
    print(f"  results/comparison_table.csv  — plain model vs elen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinanceBench evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all 150)",
    )
    args = parser.parse_args()
    run_benchmark(limit=args.limit)