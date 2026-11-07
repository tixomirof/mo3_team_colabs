"""Общее ядро RAG для ноутбука ML-HW10 и web-чата."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import httpx
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = Path(__file__).resolve().parent
LAB_DATA_DIR = LAB_DIR / "data"
# отобранные 10–20 резюме рядом с лабой; fallback — полный датасет
SELECTED_CSV = LAB_DATA_DIR / "resumes_selected.csv"
SOURCE_CSV = ROOT / "data" / "resumes_ru_6000_labeled.csv"
DATA_CSV = SELECTED_CSV if SELECTED_CSV.exists() else SOURCE_CSV
# ASCII-путь: FAISS на Windows ломается на кириллице в пути
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", str(Path.home() / ".cache" / "mo3_ml_hw10_rag")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
OLLAMA_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"))

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NO_INFO_RU = (
    "В базе нет релевантной информации по этому запросу "
    "(нет такого кандидата / категории / факта)."
)
_NO_INFO_MARKERS = (
    "в базе нет релевантной",
    "нет релевантной информации",
    "нет информации по этому запросу",
    "информации нет",
)


@dataclass
class GenParams:
    temperature: float = 0.45
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 140


_tokenizer = None
_embed_model = None


def get_embedder():
    global _tokenizer, _embed_model
    if _tokenizer is None:
        # без jupyter-widgets прогресс-баров в выводе ноутбука
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:
            pass
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _embed_model.eval()
    return _tokenizer, _embed_model


def embed_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    tokenizer, model = get_embedder()
    vectors = []
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            vectors.append(_embed_batch(batch, tokenizer, model))
            batch = []
    if batch:
        vectors.append(_embed_batch(batch, tokenizer, model))
    return np.vstack(vectors).astype("float32")


def _embed_batch(batch: list[str], tokenizer, model) -> np.ndarray:
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()


def build_faiss(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def is_relevant(distance: float, threshold: float = 40.0) -> bool:
    return float(distance) < threshold


def extract_candidate_ids(query: str) -> list[int]:
    """Достаёт номера кандидатов из вопроса: «кандидат 1», «№1», «кандидата №3»."""
    patterns = [
        r"кандидат(?:а|у|ом|е)?\s*№?\s*(\d+)",
        r"№\s*(\d+)",
        r"\bno\.?\s*(\d+)\b",
        r"\bid\s*[:=]?\s*(\d+)\b",
    ]
    ids: list[int] = []
    for pat in patterns:
        for m in re.finditer(pat, query, flags=re.IGNORECASE):
            cid = int(m.group(1))
            if cid not in ids:
                ids.append(cid)
    return ids


def find_docs_by_candidate_id(documents: list[str], candidate_id: int) -> list[str]:
    needle = f"Кандидат №{candidate_id}"
    return [doc for doc in documents if doc.startswith(needle + "\n") or doc.startswith(needle + " ")]


_SEARCH_STOP_PREFIX = re.compile(
    r"^(?:"
    r"сколько|какие|какой|какая|какое|каких|каким|какими|"
    r"кто|кого|кому|кем|"
    r"что|чем|чего|где|когда|как|"
    r"есть\s+ли|имеется\s+ли|"
    r"у\s+каких|у\s+кого|"
    r"перечисли|назови|найди|подбери"
    r")\b[\s,.:;−–—-]*",
    re.IGNORECASE,
)


def search_text(query: str) -> str:
    """Только для FAISS: предмет поиска без служебной вопросительной оболочки.
    На ответ это не влияет — в модель уходит полный исходный вопрос."""
    q = (query or "").strip()
    for _ in range(3):
        nxt = _SEARCH_STOP_PREFIX.sub("", q).strip()
        if nxt == q:
            break
        q = nxt
    q = q.strip(" ?!.…")
    return q or query.strip()


def retrieve(
    query: str,
    index: faiss.Index,
    documents: list[str],
    top_k: int = 3,
) -> tuple[list[str], np.ndarray]:
    """Явный № кандидата + семантика по тексту резюме. Категории не используем.

    В индекс идём по search_text(query), чтобы «какие …» и «сколько …»
    находили одни и те же резюме. Формулировка вопроса для ответа не меняется.
    """
    id_docs: list[str] = []
    for cid in extract_candidate_ids(query):
        id_docs.extend(find_docs_by_candidate_id(documents, cid))

    q = embed_texts([search_text(query)])
    distances, indices = index.search(q, max(top_k, 5))
    semantic_docs = [documents[i] for i in indices[0] if i >= 0]
    semantic_dists = [float(x) for x in distances[0]]

    docs: list[str] = []
    dists: list[float] = []

    def _add(doc: str, dist: float):
        if doc in docs:
            return
        docs.append(doc)
        dists.append(dist)

    for doc in id_docs:
        _add(doc, 0.0)
    for doc, dist in zip(semantic_docs, semantic_dists):
        _add(doc, dist)

    limit = max(top_k, len(id_docs), 5)
    return docs[:limit], np.asarray(dists[:limit], dtype="float32")


def ollama_generate(
    prompt: str,
    params: GenParams | None = None,
    system: str | None = None,
    model: str | None = None,
) -> str:
    params = params or GenParams()
    model = model or OLLAMA_MODEL
    system = system or (
        "Ты ассистент по базе резюме. Отвечай на вопрос только по тексту резюме в контексте. "
        "Игнорируй строку «Категория: ...», если она есть. "
        "Отвечай прямо по формулировке вопроса (число, список, да/нет — как спрашивают), "
        "плюс короткое почему по фактам. 1–2 предложения. "
        "Если точной роли нет, но опыт близкий — предложи таких кандидатов; "
        "отказ только если совсем мимо. Не выдумывай. Либо ответ, либо отказ — не оба."
    )
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "num_predict": params.max_new_tokens,
        },
    }
    base = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", OLLAMA_URL)).rstrip("/")
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _NO_INFO_MARKERS)


def normalize_answer(text: str) -> str:
    """Либо отказ, либо факт — без гибрида «нет информации, но вот кандидат…»."""
    text = (text or "").strip()
    if not text:
        return NO_INFO_RU
    if not _is_refusal(text):
        return text

    # вырезаем фразы отказа и смотрим, остался ли отдельный ответ
    rest = text
    rest = re.sub(
        r"в базе нет релевантной информации[^.!?\n]*[.!?]?",
        " ",
        rest,
        flags=re.IGNORECASE,
    )
    rest = re.sub(
        r"нет релевантной информации[^.!?\n]*[.!?]?",
        " ",
        rest,
        flags=re.IGNORECASE,
    )
    rest = re.sub(
        r"\(нет такого кандидата\s*/\s*категории\s*/\s*факта\)\.?",
        " ",
        rest,
        flags=re.IGNORECASE,
    )
    rest = re.sub(r"^\s*(однако|но|хотя|тем не менее|всё же|все же)[,:\s]+", "", rest, flags=re.I)
    rest = re.sub(r"\s+", " ", rest).strip(" \n\t-—;:,.")

    # остался содержательный ответ про кандидатов — берём только его
    if rest and re.search(r"кандидат|категори|опыт|резюме", rest, flags=re.I):
        return rest
    return NO_INFO_RU


def rag_answer(
    query: str,
    index: faiss.Index,
    documents: list[str],
    top_k: int = 3,
    distance_threshold: float = 40.0,
    params: GenParams | None = None,
    lang: str = "ru",
    model: str | None = None,
) -> dict:
    asked_ids = extract_candidate_ids(query)
    # Явно спросили про номер, которого нет в базе — сразу отказ, без LLM
    if asked_ids:
        missing = [cid for cid in asked_ids if not find_docs_by_candidate_id(documents, cid)]
        if missing and len(missing) == len(asked_ids):
            return {
                "answer": NO_INFO_RU,
                "retrieved": [],
                "relevant": [],
                "distances": [],
                "used_llm": False,
            }

    retrieved, distances = retrieve(query, index, documents, top_k=top_k)
    relevant = [
        doc for doc, dist in zip(retrieved, distances) if is_relevant(dist, distance_threshold)
    ]
    if not relevant:
        return {
            "answer": NO_INFO_RU,
            "retrieved": retrieved,
            "relevant": [],
            "distances": [float(x) for x in distances],
            "used_llm": False,
        }

    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(relevant))
    id_hint = ""
    if asked_ids:
        id_hint = (
            f" В вопросе указан кандидат №{', №'.join(map(str, asked_ids))}. "
            "Отвечай ТОЛЬКО про этого кандидата, не подменяй номер."
        )
    prompt = (
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответь прямо на вопрос в 1–2 предложениях по тексту опыта "
        "(не обходи формулировку вопроса стороной). "
        "Если роли нет дословно — ближайшие по опыту."
        f"{id_hint}\n"
        "Формат — одно из двух:\n"
        "1) короткий ответ с обоснованием;\n"
        f"2) ровно: {NO_INFO_RU}\n"
        "Не совмещай отказ и ответ."
    )
    answer = normalize_answer(ollama_generate(prompt, params=params, model=model))
    return {
        "answer": answer,
        "retrieved": retrieved,
        "relevant": relevant,
        "distances": [float(x) for x in distances],
        "used_llm": True,
    }


def anonymize_text(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", "[phone]", text)
    return text.strip()


def sample_resumes(n: int = 18, seed: int = 42) -> pd.DataFrame:
    """По заданию 10–20 обезличенных резюме. Равномерно по категориям из полного CSV."""
    df = pd.read_csv(SOURCE_CSV)
    per_class = max(1, n // df["class"].nunique())
    parts = []
    for _, g in df.groupby("class", sort=True):
        parts.append(g.sample(n=min(per_class, len(g)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        rest = df.loc[~df.index.isin(out.index)]
        need = n - len(out)
        if len(rest) > 0:
            out = pd.concat(
                [out, rest.sample(n=min(need, len(rest)), random_state=seed)],
                ignore_index=True,
            )
    out = out.head(n).reset_index(drop=True)
    out["candidate_id"] = out.index + 1
    return out


def load_selected_frame() -> pd.DataFrame:
    """Читает уже сохранённую выборку из labs/ML-HW10/data/."""
    if not SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Нет {SELECTED_CSV}. Сначала запустите prepare_data.py"
        )
    df = pd.read_csv(SELECTED_CSV)
    if "candidate_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["candidate_id"] = df.index + 1
    return df


def load_resume_documents(
    with_tags: bool = True,
    n: int = 18,
    seed: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    if SELECTED_CSV.exists():
        df = load_selected_frame().head(n).reset_index(drop=True)
    else:
        df = sample_resumes(n=n, seed=seed)
    docs = []
    for _, row in df.iterrows():
        raw = anonymize_text(str(row["text"]).replace("\r", "\n"))
        cid = int(row["candidate_id"])
        label = str(row["class"])
        if with_tags:
            docs.append(f"Кандидат №{cid}\nКатегория: {label}\n{raw}")
        else:
            docs.append(f"Кандидат №{cid}\n{raw}")
    return docs, df


def _faiss_write(index: faiss.Index, dest: Path) -> None:
    """FAISS на Windows плохо пишет по Unicode-путям — через %TEMP%."""
    import tempfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"faiss_{os.getpid()}_{dest.name}"
    faiss.write_index(index, str(tmp))
    dest.write_bytes(tmp.read_bytes())
    tmp.unlink(missing_ok=True)


def _faiss_read(src: Path) -> faiss.Index:
    import tempfile

    tmp = Path(tempfile.gettempdir()) / f"faiss_read_{os.getpid()}_{src.name}"
    tmp.write_bytes(src.read_bytes())
    try:
        return faiss.read_index(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def save_index(name: str, documents: list[str], embeddings: np.ndarray, index: faiss.Index) -> Path:
    folder = CACHE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", embeddings)
    (folder / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False), encoding="utf-8"
    )
    _faiss_write(index, folder / "index.faiss")
    return folder


def load_index(name: str) -> tuple[list[str], faiss.Index] | None:
    folder = CACHE_DIR / name
    if not (folder / "index.faiss").exists():
        return None
    index = _faiss_read(folder / "index.faiss")
    documents = json.loads((folder / "documents.json").read_text(encoding="utf-8"))
    return documents, index


def ensure_resume_indices(force: bool = False, n: int = 18) -> dict:
    result = {}
    for name, with_tags in [("resumes_with_tags", True), ("resumes_no_tags", False)]:
        if not force:
            loaded = load_index(name)
            if loaded is not None and len(loaded[0]) == n:
                docs, index = loaded
                result[name] = {"documents": docs, "index": index, "n": n}
                continue
        docs, meta = load_resume_documents(with_tags=with_tags, n=n)
        emb = embed_texts(docs, batch_size=16)
        index = build_faiss(emb)
        save_index(name, docs, emb, index)
        meta_path = CACHE_DIR / name / "meta.csv"
        meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
        result[name] = {"documents": docs, "index": index, "n": n, "meta": meta}
    return result
