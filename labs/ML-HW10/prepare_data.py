"""Отбирает 18 резюме и кладёт в labs/ML-HW10/data/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_system import anonymize_text, sample_resumes

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
N = 18


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = sample_resumes(n=N, seed=42)
    df["text"] = df["text"].map(lambda t: anonymize_text(str(t).replace("\r", "\n")))
    out_csv = DATA_DIR / "resumes_selected.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # по одному txt на кандидата — удобно смотреть глазами
    txt_dir = DATA_DIR / "raw"
    txt_dir.mkdir(exist_ok=True)
    for _, row in df.iterrows():
        cid = int(row["candidate_id"])
        label = str(row["class"]).replace(" ", "_")
        path = txt_dir / f"candidate_{cid:02d}_{label}.txt"
        path.write_text(
            f"Кандидат №{cid}\nКатегория: {row['class']}\n\n{row['text']}",
            encoding="utf-8",
        )

    print("saved", out_csv, "rows=", len(df))
    print(df[["candidate_id", "class"]].to_string(index=False))
    print("raw texts:", txt_dir)


if __name__ == "__main__":
    main()
