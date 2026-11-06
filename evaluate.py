"""
Run every question in questions.csv two times:
  (1) WITH documents (RAG)   (2) WITHOUT documents
Saves results.csv so the team can score each answer.

Run:  python evaluate.py              (default model)
      python evaluate.py qwen2.5:3b   (other model)
"""
import csv
import sys

from rag import DEFAULT_MODEL, Retriever, answer, load_chunks

model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
retriever = Retriever(load_chunks())

with open("questions.csv", encoding="utf-8") as f:
    questions = list(csv.DictReader(f))

rows = []
for q in questions:
    print(f"[{q['id']}/{len(questions)}] {q['question']}")
    with_docs, hits = answer(q["question"], True, retriever, model)
    without_docs, _ = answer(q["question"], False, retriever, model)
    rows.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "answer_with_docs": with_docs,
        "answer_without_docs": without_docs,
        "sources": "; ".join(sorted({h["source"] for h in hits})),
        # Fill these in by hand: 2 = correct, 1 = partly correct, 0 = wrong / made up
        "score_with_docs": "",
        "score_without_docs": "",
    })

# utf-8-sig so Thai text opens correctly in Excel
with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Saved {len(rows)} rows to results.csv (model: {model})")
print("Next: open results.csv, give each answer a score (2/1/0), then compare the average per category.")
