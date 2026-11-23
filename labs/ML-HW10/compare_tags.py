"""ML-HW10: честное сравнение тегированной и нетегированной базы.

Обе базы (18 резюме) собираются в памяти, существующие индексы в кэше не трогаются.
На обеих прогоняется один и тот же набор настроек из run_experiments.run_suite
и одни и те же вопросы. Результаты пишутся в отдельную папку (TAGS_OUT).

Запуск: python compare_tags.py  (из папки labs/ML-HW10)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from questions import QUESTIONS  # noqa: E402
from rag_system import (  # noqa: E402
    CACHE_DIR, NO_INFO_RU, build_faiss, embed_texts, extract_candidate_ids, load_resume_documents,
    retrieve, search_text, _is_refusal,
)
from run_experiments import run_suite  # noqa: E402

OUT = Path(os.getenv("TAGS_OUT", str(CACHE_DIR / "compare_tags")))
THRESHOLD = 55.0

# Дополнительные вопросы о сфере деятельности: на них тег может влиять сильнее всего.
# В выборке из 18 резюме по три кандидата на категорию (см. resumes_selected.csv).
EXTRA = [
    {"q": "Кто из кандидатов работает в сфере образования и науки?", "kind": "category",
     "expected": "Кандидат №4, №5, №6", "ids": [4, 5, 6]},
    {"q": "Какие кандидаты связаны с транспортом и логистикой?", "kind": "category",
     "expected": "Кандидат №13, №14, №15", "ids": [13, 14, 15]},
    {"q": "Есть ли в базе специалисты по финансам и бухгалтерии?", "kind": "category",
     "expected": "Кандидат №16, №17, №18", "ids": [16, 17, 18]},
]


# Три показательных вопроса, подобранные перебором ~20 формулировок «кто из сферы X»
# на обеих базах (baseline-настройки). Ожидаемые номера — по разметке class.
SHOWCASE = [
    {"q": "Есть ли в базе специалисты по финансам и бухгалтерии?", "kind": "category",
     "expected": "Кандидат №16, №17, №18", "ids": [16, 17, 18]},
    {"q": "Кого можно взять на работу в поликлинику?", "kind": "category",
     "expected": "Кандидат №3 (медицинский представитель)", "ids": [3]},
    {"q": "Кто из кандидатов работает в медицине и здравоохранении?", "kind": "category",
     "expected": "По разметке №1, №2, №3; по тексту резюме к медицине относится только №3", "ids": [3]},
]


def showcase():
    """Топ-3 поиска и ответ на baseline-настройках для SHOWCASE на обеих базах."""
    from rag_system import GenParams, rag_answer
    OUT.mkdir(parents=True, exist_ok=True)
    base = GenParams(temperature=0.1, top_k=50, top_p=0.8, max_new_tokens=100)
    rows = []
    for name, tags in (("resumes_with_tags", True), ("resumes_no_tags", False)):
        docs, _ = load_resume_documents(with_tags=tags, n=18)
        index = build_faiss(embed_texts(docs, batch_size=16))
        for q in SHOWCASE:
            hits, dists = retrieve(q["q"], index, docs, top_k=3)
            t0 = time.time()
            out = rag_answer(q["q"], index, docs, top_k=3, distance_threshold=THRESHOLD, params=base, lang="ru")
            top = [(int(re.match(r"Кандидат №(\d+)", d).group(1)), float(x)) for d, x in zip(hits, dists)]
            rows.append({
                "dataset": name, "question": q["q"], "expected": q["expected"],
                "top3": ", ".join(f"№{n} ({x:.2f})" for n, x in top[:3]),
                "n_relevant": len(out["relevant"]), "answer": out["answer"],
                "sec": round(time.time() - t0, 2),
            })
            print(name, "|", q["q"], "|", rows[-1]["top3"], "|", out["answer"][:120], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "showcase_tags.csv", index=False, encoding="utf-8-sig")
    print("Saved", OUT / "showcase_tags.csv")


def expected_ids(q):
    if "ids" in q:
        return q["ids"]
    return [int(x) for x in re.findall(r"№\s*(\d+)", q.get("expected", ""))]


def mentioned(answer):
    return {int(x) for x in re.findall(r"(?:№|кандидат(?:а|у|ом)?\s*)\s*(\d+)", answer.lower())}


def score(q, answer):
    """0–2 по той же шкале, что в ML-HW11: 2 — полный ответ/обоснованный отказ, 1 — частично, 0 — мимо."""
    refusal = _is_refusal(answer) or answer.strip() == NO_INFO_RU
    if q["kind"] == "no_answer":
        return 2 if refusal else 0
    ids = set(expected_ids(q))
    hits = ids & mentioned(answer)
    if refusal and not hits:
        return 0
    if hits == ids:
        return 2
    return 1 if hits else 0


def retrieval_rows(name, docs, index, questions):
    rows = []
    for q in questions:
        ids = expected_ids(q)
        hits, dists = retrieve(q["q"], index, docs, top_k=10)
        found = [int(re.match(r"Кандидат №(\d+)", d).group(1)) for d in hits]
        for cid in ids or [None]:
            rank = found.index(cid) + 1 if cid in found else None
            rows.append({
                "dataset": name, "question": q["q"], "kind": q["kind"], "expected_id": cid,
                "rank_top10": rank,
                "distance": round(float(dists[found.index(cid)]), 2) if rank else None,
                "in_top3": bool(rank and rank <= 3),
                "relevant_at_55": bool(rank and rank <= 3 and dists[found.index(cid)] <= THRESHOLD),
                "top3": ", ".join(f"№{x}" for x in found[:3]),
                "top1_distance": round(float(dists[0]), 2),
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions = [dict(q) for q in QUESTIONS] + EXTRA
    packs = {}
    for name, tags in (("resumes_with_tags", True), ("resumes_no_tags", False)):
        t0 = time.time()
        docs, _ = load_resume_documents(with_tags=tags, n=18)
        index = build_faiss(embed_texts(docs, batch_size=16))
        packs[name] = {"documents": docs, "index": index, "build_sec": round(time.time() - t0, 2)}
        print(name, len(docs), "docs", packs[name]["build_sec"], "s", flush=True)

    retrieval = []
    for name, pack in packs.items():
        retrieval += retrieval_rows(name, pack["documents"], pack["index"], questions)
    pd.DataFrame(retrieval).to_csv(OUT / "retrieval_tags_vs_no_tags.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name, pack in packs.items():
        print("Suite on", name, flush=True)
        rows += run_suite(name, pack, questions, threshold=THRESHOLD)
    kinds = {q["q"]: q["kind"] for q in questions}
    by_q = {q["q"]: q for q in questions}
    for row in rows:
        row["kind"] = kinds[row["question"]]
        row["score"] = score(by_q[row["question"]], row["answer"])
        row["homework"] = row["question"] in {QUESTIONS[i]["q"] for i in (0, 1, 3)}
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "experiments_tags_vs_no_tags.csv", index=False, encoding="utf-8-sig")
    summary = (frame.groupby(["settings", "dataset"])["score"].mean().unstack("dataset").round(2))
    summary.to_csv(OUT / "summary_by_settings.csv", encoding="utf-8-sig")
    print(summary.to_string(), flush=True)
    (OUT / "build.json").write_text(json.dumps(
        {k: v["build_sec"] for k, v in packs.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Saved to", OUT)


if __name__ == "__main__":
    if sys.argv[1:] == ["showcase"]:
        showcase()
    else:
        main()
