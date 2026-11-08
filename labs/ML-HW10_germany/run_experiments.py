"""Эксперименты: top_k поиска, temperature, top_k/top_p генерации."""

from __future__ import annotations

import time

import pandas as pd

from questions import HOMEWORK_QUESTIONS
from rag_system import CACHE_DIR, GEN_MODEL_NAME, GenParams, ensure_berlin_index, rag_answer

OUT = CACHE_DIR / "experiments_berlin.csv"
DISTANCE_THRESHOLD = 40.0


def evaluate_answer(kind: str, answer: str, expected: str, keywords: list[str] | None = None) -> str:
    a = (answer or "").lower()
    if kind == "no_answer":
        markers = [
            "нет релевантной",
            "нет информации",
            "в базе нет",
            "не найден",
            "отсутств",
            "нет такого",
            "no relevant",
        ]
        return "ok: отказ без галлюцинации" if any(m in a for m in markers) else "плохо: возможная галлюцинация"

    keys = [k.lower() for k in (keywords or []) if k]
    if expected:
        keys.append(expected.lower())
    hits = sum(1 for k in keys if k and k in a)
    if hits >= 1:
        return "ok: есть ожидаемый факт"
    words = [w for w in expected.lower().replace("/", " ").split() if len(w) > 2]
    if words and sum(1 for w in words if w in a) >= max(1, len(words) // 2):
        return "частично"
    return "слабо"


def run_suite(pack: dict, questions: list[dict]) -> list[dict]:
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
            distance_threshold=DISTANCE_THRESHOLD,
            params=params,
        )
        rows.append(
            {
                "dataset": "berlin_facts",
                "question": q["q"],
                "expected": q.get("expected", ""),
                "settings": settings,
                "answer": out["answer"],
                "distances": out["distances"],
                "n_relevant": len(out["relevant"]),
                "eval": evaluate_answer(
                    q["kind"], out["answer"], q.get("expected", ""), q.get("keywords")
                ),
                "sec": round(time.time() - t0, 2),
                "model": GEN_MODEL_NAME,
            }
        )
        print(f"  [{settings[:48]}] {q['q'][:40]}... -> {rows[-1]['eval']} ({rows[-1]['sec']}s)")

    print("baseline...")
    for q in questions:
        one(q, "baseline top_k_search=3 temp=0.1 top_k_gen=50 top_p=0.8")

    print("top_k_search...")
    for top_k in [1, 3, 5, 10]:
        for q in questions:
            one(q, f"top_k_search={top_k} (остальное baseline)", top_k_search=top_k)

    print("temperature (3 repeats)...")
    q0 = questions[0]
    for temp in [0.1, 0.4, 0.7, 1.0]:
        for rep in range(1, 4):
            one(
                q0,
                f"temperature={temp} repeat={rep}",
                params=GenParams(temperature=temp, top_k=50, top_p=0.8, max_new_tokens=100),
            )

    print("top_k_gen...")
    for top_k_gen in [5, 10, 50, 100]:
        for q in questions:
            one(
                q,
                f"top_k_gen={top_k_gen}",
                params=GenParams(temperature=0.1, top_k=top_k_gen, top_p=0.8, max_new_tokens=100),
            )

    print("top_p...")
    for top_p in [0.3, 0.5, 0.8, 1.0]:
        for q in questions:
            one(
                q,
                f"top_p={top_p}",
                params=GenParams(temperature=0.1, top_k=50, top_p=top_p, max_new_tokens=100),
            )

    return rows


def main():
    print(f"Building Berlin FAISS index; model={GEN_MODEL_NAME}")
    pack = ensure_berlin_index(force=False)
    print("documents:", pack["n"])

    # прогрев генератора один раз
    from rag_system import get_generator

    get_generator()
    print("generator ready")

    rows = run_suite(pack, HOMEWORK_QUESTIONS)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("Saved", OUT, "rows=", len(df))
    print(df.groupby(df["settings"].str.split().str[0])["eval"].apply(
        lambda s: s.astype(str).str.startswith("ok").mean()
    ))


if __name__ == "__main__":
    main()
