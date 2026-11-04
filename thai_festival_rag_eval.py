#!/usr/bin/env python3
"""
Thai festivals QA: does giving an AI local documents improve accuracy?

Conditions
  baseline : the model answers from its own knowledge
  rag      : the model gets the top-k retrieved chunks from ./docs

Metrics (per question)
  fact_recall : share of required fact groups found in the answer (deterministic)
  judge_score : 0 / 0.5 / 1 from a blind LLM judge comparing against the reference answer
  retrieval_hit (rag only) : share of required fact groups present in retrieved chunks

Setup
  pip install anthropic scikit-learn pandas numpy
  export ANTHROPIC_API_KEY=...
  put .txt / .md files (Thai or English) in ./docs
  edit questions.json so the reference answers and key facts match YOUR documents

Run
  python thai_festival_rag_eval.py --docs docs --questions questions.json --k 4
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEFAULT_MODEL = "claude-sonnet-4-6"


# ----------------------------------------------------------------- utilities
def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text)


def load_chunks(doc_dir: str, max_chars: int = 900, overlap: int = 150):
    """Paragraph-aware chunking. Returns list of {source, text}."""
    chunks = []
    for path in sorted(Path(doc_dir).rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) > max_chars:
                chunks.append({"source": path.name, "text": buf})
                buf = buf[-overlap:] + "\n" + p
            else:
                buf = (buf + "\n\n" + p) if buf else p
        if buf:
            chunks.append({"source": path.name, "text": buf})
    if not chunks:
        raise SystemExit(f"No .txt/.md documents found in {doc_dir}")
    return chunks


class Retriever:
    """TF-IDF over character n-grams: works for Thai (no spaces) and English."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([norm(c["text"]) for c in chunks])

    def search(self, query: str, k: int):
        sims = linear_kernel(self.vec.transform([norm(query)]), self.matrix).ravel()
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]


# ----------------------------------------------------------------- model calls
BASE_SYSTEM = (
    "You are an expert on Thai festivals and traditions. Answer the question accurately "
    "and concisely (max 120 words). If you are not sure, say so rather than guessing."
)
RAG_SYSTEM = (
    "You are an expert on Thai festivals and traditions. Use the provided documents as "
    "your primary source. Answer accurately and concisely (max 120 words). If the "
    "documents do not contain the answer, say so rather than guessing."
)

JUDGE_SYSTEM = (
    "You are a strict grader. Compare the candidate answer with the reference answer. "
    'Reply with JSON only: {"score": 0 | 0.5 | 1, "reason": "<one sentence>"}. '
    "1 = all key facts correct and no contradictions; 0.5 = partly correct or missing "
    "key facts; 0 = wrong, contradicts the reference, or refuses to answer."
)


def ask(client, model, system, user, max_tokens=400):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def answer_baseline(client, model, question):
    return ask(client, model, BASE_SYSTEM, question)


def answer_rag(client, model, question, retrieved):
    ctx = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved)
    user = f"Documents:\n{ctx}\n\nQuestion: {question}"
    return ask(client, model, RAG_SYSTEM, user)


def judge(client, model, question, reference, candidate):
    user = (
        f"Question: {question}\n\nReference answer: {reference}\n\n"
        f"Candidate answer: {candidate}"
    )
    out = ask(client, model, JUDGE_SYSTEM, user, max_tokens=200)
    m = re.search(r'"score"\s*:\s*(0\.5|0|1)', out)
    return float(m.group(1)) if m else np.nan


# ----------------------------------------------------------------- scoring
def fact_recall(text: str, fact_groups):
    """fact_groups: list of lists; each inner list = acceptable spellings/variants."""
    t = norm(text)
    hits = [any(norm(alt) in t for alt in group) for group in fact_groups]
    return sum(hits) / len(hits) if hits else np.nan


def bootstrap_ci(diffs, n=5000, seed=0):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return diffs.mean(), *np.percentile(means, [2.5, 97.5])


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--k", type=int, default=4, help="chunks retrieved per question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--judge-model", default=None, help="defaults to --model")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    judge_model = args.judge_model or args.model

    client = anthropic.Anthropic()
    retriever = Retriever(load_chunks(args.docs))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    rows = []
    for q in questions:
        retrieved = retriever.search(q["question"], args.k)
        ctx_text = " ".join(c["text"] for c in retrieved)
        answers = {
            "baseline": answer_baseline(client, args.model, q["question"]),
            "rag": answer_rag(client, args.model, q["question"], retrieved),
        }
        for cond, ans in answers.items():
            rows.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "festival": q.get("festival", ""),
                    "condition": cond,
                    "question": q["question"],
                    "answer": ans,
                    "fact_recall": fact_recall(ans, q["key_facts"]),
                    "judge_score": judge(client, judge_model, q["question"], q["reference"], ans),
                    "retrieval_hit": fact_recall(ctx_text, q["key_facts"]) if cond == "rag" else np.nan,
                    "sources": ", ".join(sorted({c["source"] for c in retrieved})) if cond == "rag" else "",
                }
            )
        print(f"done {q['id']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # ---- summaries
    print("\n=== Mean scores by condition ===")
    print(df.groupby("condition")[["fact_recall", "judge_score"]].mean().round(3))

    print("\n=== By category ===")
    print(df.pivot_table(index="category", columns="condition",
                         values=["fact_recall", "judge_score"], aggfunc="mean").round(3))

    print("\n=== By festival ===")
    print(df.pivot_table(index="festival", columns="condition",
                         values="judge_score", aggfunc="mean").round(3))

    print("\n=== Paired difference (rag - baseline), 95% bootstrap CI ===")
    for metric in ["fact_recall", "judge_score"]:
        wide = df.pivot(index="id", columns="condition", values=metric)
        mean, lo, hi = bootstrap_ci(wide["rag"] - wide["baseline"])
        print(f"{metric:12s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(wide)})")

    print("\n=== RAG diagnostics ===")
    r = df[df.condition == "rag"]
    print(f"mean retrieval_hit: {r.retrieval_hit.mean():.3f}")
    miss = r[(r.retrieval_hit < 1) & (r.judge_score < 1)]
    print(f"questions where retrieval missed facts AND answer was imperfect: {len(miss)}")
    print(f"\nPer-question results saved to {args.out}")


if __name__ == "__main__":
    main()
