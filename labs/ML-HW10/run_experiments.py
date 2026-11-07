"""10–20 резюме, 2 FAISS-базы, эксперименты параметров."""

from __future__ import annotations

import time

import pandas as pd

from questions import HOMEWORK_QUESTIONS
from rag_system import (
    CACHE_DIR,
    GenParams,
    OLLAMA_MODEL,
    ensure_resume_indices,
    rag_answer,
)

OUT = CACHE_DIR / "experiments_resumes.csv"
N_RESUMES = 18


def evaluate_answer(kind: str, answer: str, expected: str) -> str:
    a = (answer or "").lower().replace("№", "№").replace("no", "№")
    a_norm = a.replace(" ", "")
    if kind == "no_answer":
        markers = [
            "нет релевантной",
            "нет информации",
            "в базе нет",
            "такой категории нет",
            "не найден",
            "отсутств",
            "нет такого",
            "не указан",
        ]
        return "ok: отказ без галлюцинации" if any(m in a for m in markers) else "плохо: возможная галлюцинация"
    if expected:
        exp = expected.lower().replace(" ", "")
        # «Кандидат №1» / «кандидат 1» / «№1»
        digits = "".join(ch for ch in expected if ch.isdigit())
        if exp in a_norm or (digits and (f"№{digits}" in a or f"кандидат№{digits}" in a_norm or f"кандидат {digits}" in a)):
            return "ok: есть ожидаемый факт"
        words = [w for w in expected.lower().replace("/", " ").split() if len(w) > 2]
        hits = sum(1 for w in words if w in a)
        if hits >= max(1, len(words) // 2):
            return "частично"
        return "слабо"
    return "проверено"


def run_suite(dataset: str, pack: dict, questions: list[dict], threshold: float) -> list[dict]:
    rows = []
    docs, index = pack["documents"], pack["index"]
    base = GenParams(temperature=0.1, top_k=50, top_p=0.8, max_new_tokens=100)

    def one(q, settings, top_k_search=3, params=None):
        params = params or base
        t0 = time.time()
        out = rag_answer(
            q["q"],
            index,
            docs,
            top_k=top_k_search,
            distance_threshold=threshold,
            params=params,
            lang="ru",
        )
        rows.append(
            {
                "dataset": dataset,
                "question": q["q"],
                "expected": q.get("expected", ""),
                "settings": settings,
                "answer": out["answer"],
                "distances": out["distances"],
                "n_relevant": len(out["relevant"]),
                "eval": evaluate_answer(q["kind"], out["answer"], q.get("expected", "")),
                "sec": round(time.time() - t0, 2),
                "model": OLLAMA_MODEL,
            }
        )
        print(f"  [{settings[:40]}] {q['q'][:50]}... -> {rows[-1]['eval']} ({rows[-1]['sec']}s)")

    for q in questions:
        one(q, "baseline top_k_search=3 temp=0.1 top_k_gen=50 top_p=0.8")

    for top_k in [1, 3, 5, 10]:
        for q in questions:
            one(q, f"top_k_search={top_k} (остальное baseline)", top_k_search=top_k)

    q0 = questions[0]
    for temp in [0.1, 0.4, 0.7, 1.0]:
        for rep in range(1, 4):
            one(
                q0,
                f"temperature={temp} repeat={rep}",
                params=GenParams(temperature=temp, top_k=50, top_p=0.8, max_new_tokens=100),
            )

    for top_k_gen in [5, 10, 50, 100]:
        for q in questions:
            one(
                q,
                f"top_k_gen={top_k_gen}",
                params=GenParams(temperature=0.1, top_k=top_k_gen, top_p=0.8, max_new_tokens=100),
            )

    for top_p in [0.3, 0.5, 0.8, 1.0]:
        for q in questions:
            one(
                q,
                f"top_p={top_p}",
                params=GenParams(temperature=0.1, top_k=50, top_p=top_p, max_new_tokens=100),
            )

    return rows


def main():
    print(f"Building resume FAISS indices (n={N_RESUMES})...")
    packs = ensure_resume_indices(force=True, n=N_RESUMES)
    for k, v in packs.items():
        print(k, len(v["documents"]))

    questions = HOMEWORK_QUESTIONS
    print("Questions:")
    for q in questions:
        print(" -", q["kind"], q["q"], "| gold:", q.get("gold", ""))

    rows = []
    print("Experiments on resumes_with_tags...")
    rows += run_suite("resumes_with_tags", packs["resumes_with_tags"], questions, threshold=55.0)

    print("Baseline comparison on resumes_no_tags...")
    docs = packs["resumes_no_tags"]["documents"]
    index = packs["resumes_no_tags"]["index"]
    base = GenParams(temperature=0.1, top_k=50, top_p=0.8, max_new_tokens=100)
    for q in questions:
        t0 = time.time()
        out = rag_answer(q["q"], index, docs, top_k=3, distance_threshold=55.0, params=base, lang="ru")
        rows.append(
            {
                "dataset": "resumes_no_tags",
                "question": q["q"],
                "expected": q.get("expected", ""),
                "settings": "baseline top_k_search=3 temp=0.1 top_k_gen=50 top_p=0.8",
                "answer": out["answer"],
                "distances": out["distances"],
                "n_relevant": len(out["relevant"]),
                "eval": evaluate_answer(q["kind"], out["answer"], q.get("expected", "")),
                "sec": round(time.time() - t0, 2),
                "model": OLLAMA_MODEL,
            }
        )
        print("  no_tags:", q["q"][:50], "->", rows[-1]["eval"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("Saved", OUT, "rows=", len(df))


if __name__ == "__main__":
    main()
