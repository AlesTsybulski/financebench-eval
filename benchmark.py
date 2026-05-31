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
RAW_RESULTS_PATH      = "results/raw_results.csv"
COMPARISON_TABLE_PATH = "results/comparison_table.csv"


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


def create_judge_conversation() -> str | None:
    body = {
        "title": "judge",
        "model": MODEL_ID,
        "tools": {
            "webSearch": False,
            "codeInterpreter": False,
            "imageGeneration": False,
        }
    }
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
        print(f"  [ERROR] create_judge_conversation failed: {e}")
        return None


def llm_judge(gold: str, answer: str, question: str) -> tuple[bool, str]:
    conv_id = create_judge_conversation()
    if not conv_id:
        return False, "ERROR: could not create conversation"

    prompt = (
        f"Classify whether the model answer matches the gold answer.\n\n"
        f"Reply in this exact format — the very first line must be the verdict:\n"
        f"VERDICT: YES\n"
        f"or\n"
        f"VERDICT: NO\n\n"
        f"Do not write anything before the word VERDICT. You may explain after.\n\n"
        f"Rules:\n"
        f"- Numbers that differ by less than 5% are equivalent (rounding is acceptable).\n"
        f"- A negative sign on a financial outflow amount is equivalent to the positive (e.g. -1577 and 1577 are the same).\n"
        f"- Partial answers that omit key information from the gold are NOT equivalent.\n\n"
        f"Question: {question}\n"
        f"Gold: {gold}\n"
        f"Model: {answer}\n\n"
        f"VERDICT:"
    )

    try:
        response = requests.post(
            f"{BASE_URL}/conversations/{conv_id}/messages",
            headers=HEADERS,
            json={"message": prompt},
            timeout=60,
        )
        response.raise_for_status()
        raw_reply = response.json()["assistantMessage"]["content"].strip()

        match = re.search(r"VERDICT:\s*(YES|NO)", raw_reply.upper())
        if match:
            verdict = match.group(1) == "YES"
        else:
            verdict = False
            print(f"  [WARN] llm_judge: could not parse VERDICT from: {raw_reply[:80]!r}")

        return verdict, raw_reply
    except Exception as e:
        print(f"  [ERROR] llm_judge failed: {e}")
        return False, f"ERROR: {e}"
    finally:
        delete_conversation(conv_id)


def is_correct(gold: str, answer: str, question: str) -> tuple[bool, str]:
    if pd.isnull(answer) or not answer:
        return False, "no answer"
    return llm_judge(gold, answer, question)




def load_checkpoint() -> tuple[list, set]:
    if os.path.exists(RAW_RESULTS_PATH) and os.path.getsize(RAW_RESULTS_PATH) > 0:
        df = pd.read_csv(RAW_RESULTS_PATH)
        done_ids = set(df["financebench_id"].astype(str).tolist())
        results  = df.to_dict("records")
        print(f"Checkpoint found — {len(done_ids)} questions already done, resuming...")
        return results, done_ids
    return [], set()


def run_benchmark(limit: int = None):
    rows = []
    with open("data/financebench_open_source.jsonl", "r") as f:
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
    print(f"Elen agent      : {ELEN_AGENT_ID}")
    print(f"\nThis pass collects raw answers only.")
    print(f"Run with --judge afterwards to evaluate them.\n")

    if not remaining:
        print("All questions already processed!")
        return

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

    print(f"\nCollection complete. {len(results)} rows in {RAW_RESULTS_PATH}")
    print("Run with --judge to evaluate answers.")


def run_judge(force: bool = False):
    """Read raw_results.csv, run LLM judgment for each row, write comparison_table.csv."""
    if not os.path.exists(RAW_RESULTS_PATH):
        print("No raw_results.csv found. Run benchmark collection first.")
        return

    raw_df = pd.read_csv(RAW_RESULTS_PATH)
    print(f"Loaded {len(raw_df)} rows from {RAW_RESULTS_PATH}")

    done_ids: set[str] = set()
    comparison_rows: list[dict] = []

    if not force and os.path.exists(COMPARISON_TABLE_PATH) and os.path.getsize(COMPARISON_TABLE_PATH) > 0:
        comp_df = pd.read_csv(COMPARISON_TABLE_PATH)
        if "financebench_id" in comp_df.columns:
            done_ids = set(comp_df["financebench_id"].astype(str).tolist())
            comparison_rows = comp_df.to_dict("records")
            print(f"Resuming: {len(done_ids)} rows already judged.")

    pending = raw_df[~raw_df["financebench_id"].astype(str).isin(done_ids)]
    print(f"Pending: {len(pending)} rows\n")

    if pending.empty:
        print("All rows already judged!")
    else:
        for i, row in enumerate(tqdm(pending.itertuples(index=False), total=len(pending), desc="Judging")):
            question_id = str(row.financebench_id)
            print(f"\n[{i+1}/{len(pending)}] {question_id}")

            p_correct, p_reply = is_correct(row.gold_answer, row.plain_answer, row.question)
            time.sleep(1)
            e_correct, e_reply = is_correct(row.gold_answer, row.elen_answer, row.question)
            time.sleep(1)

            comparison_rows.append({
                "financebench_id":      question_id,
                "gold_answer":          row.gold_answer,
                "Plain model":          row.plain_answer,
                "plain_is_correct":     p_correct,
                "plain_judge_response": p_reply,
                "Model + Elen":         row.elen_answer,
                "elen_is_correct":      e_correct,
                "elen_judge_response":  e_reply,
            })

            os.makedirs("results", exist_ok=True)
            pd.DataFrame(comparison_rows).to_csv(COMPARISON_TABLE_PATH, index=False)

    total       = len(comparison_rows)
    plain_score = sum(r["plain_is_correct"] for r in comparison_rows)
    elen_score  = sum(r["elen_is_correct"]  for r in comparison_rows)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Questions evaluated : {total}")
    if total:
        print(f"Plain model score   : {plain_score}/{total} ({plain_score/total*100:.1f}%)")
        print(f"Elen  model score   : {elen_score}/{total} ({elen_score/total*100:.1f}%)")
    print(f"\nFiles saved:")
    print(f"  {RAW_RESULTS_PATH:<40} — full raw answers")
    print(f"  {COMPARISON_TABLE_PATH:<40} — plain model vs elen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinanceBench evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of questions to collect (default: all 150)")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM judgment pass on raw_results.csv comparison_table.csv")
    parser.add_argument("--force-rejudge", action="store_true",
                        help="Re-judge all rows, ignoring existing comparison_table.csv")
    args = parser.parse_args()

    if args.judge:
        run_judge(force=args.force_rejudge)
    else:
        run_benchmark(limit=args.limit)