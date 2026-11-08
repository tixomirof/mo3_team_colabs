"""RAG по фактам о Берлине: FAISS + Qwen2.5 Instruct (transformers), без GPT-2."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LAB_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", str(Path.home() / ".cache" / "mo3_ml_hw10_germany")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv(
    "RAG_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# Современная instruct-модель с русским (вместо GPT-2). Не слишком тяжёлая для CPU.
GEN_MODEL_NAME = os.getenv("RAG_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

NO_INFO_RU = (
    "В базе нет релевантной информации по этому запросу "
    "(нет такого факта о Берлине / Германии)."
)
_NO_INFO_MARKERS = (
    "в базе нет релевантной",
    "нет релевантной информации",
    "нет информации по этому запросу",
    "информации нет",
    "i am sorry, there is no relevant",
    "no relevant information",
)

SYSTEM_PROMPT = (
    "Ты ассистент по базе фактов о Берлине и Германии. "
    "Отвечай только по тексту из контекста. "
    "Отвечай прямо (число, факт, да/нет — как спрашивают), 1–2 предложения на русском. "
    "Не выдумывай факты, которых нет в контексте. "
    f"Если в контексте нет ответа — напиши ровно: {NO_INFO_RU}"
)


@dataclass
class GenParams:
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.8
    max_new_tokens: int = 100


_embed_tok = None
_embed_model = None
_gen_tok = None
_gen_model = None


def get_embedder():
    global _embed_tok, _embed_model
    if _embed_tok is None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:
            pass
        _embed_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _embed_model.eval()
    return _embed_tok, _embed_model


def get_generator():
    global _gen_tok, _gen_model
    if _gen_tok is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, trust_remote_code=True)
        _gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            _gen_model = _gen_model.to("cpu")
        _gen_model.eval()
        if _gen_tok.pad_token is None:
            _gen_tok.pad_token = _gen_tok.eos_token
    return _gen_tok, _gen_model


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


def retrieve(
    query: str,
    index: faiss.Index,
    documents: list[str],
    top_k: int = 3,
) -> tuple[list[str], np.ndarray]:
    q = embed_texts([query])
    distances, indices = index.search(q, top_k)
    docs = [documents[i] for i in indices[0] if i >= 0]
    dists = np.asarray([float(x) for x in distances[0][: len(docs)]], dtype="float32")
    return docs, dists


def generate_text(
    context: str,
    query: str,
    params: GenParams | None = None,
    system: str | None = None,
) -> str:
    params = params or GenParams()
    tokenizer, model = get_generator()
    system = system or SYSTEM_PROMPT
    user = (
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответь по контексту одним-двумя предложениями на русском. "
        "Без нумерации и без повторения инструкций. "
        f"Если факта нет в контексте — напиши ровно: {NO_INFO_RU}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": params.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": max(float(params.temperature), 1e-5),
        "top_k": int(params.top_k),
        "top_p": float(params.top_p),
        "repetition_penalty": 1.1,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _NO_INFO_MARKERS)


def normalize_answer(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(
        r"^(?:\d+\)\s*)?(?:короткий ответ[^:]*:\s*|факт:\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if not text:
        return NO_INFO_RU
    if not _is_refusal(text):
        return text
    rest = re.sub(
        r"(в базе нет релевантной информации|нет релевантной информации|"
        r"i am sorry, there is no relevant information|no relevant information)[^.!?\n]*[.!?]?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    rest = re.sub(r"\([^)]*нет такого[^)]*\)", " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"^\s*(однако|но|хотя|тем не менее)[,:\s]+", "", rest, flags=re.I)
    rest = re.sub(r"\s+", " ", rest).strip(" \n\t-—;:,.")
    if rest and len(rest) > 20 and not _is_refusal(rest):
        return rest
    return NO_INFO_RU


def rag_answer(
    query: str,
    index: faiss.Index,
    documents: list[str],
    top_k: int = 3,
    distance_threshold: float = 40.0,
    params: GenParams | None = None,
) -> dict:
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

    context = "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(relevant))
    answer = normalize_answer(generate_text(context, query, params=params))
    return {
        "answer": answer,
        "retrieved": retrieved,
        "relevant": relevant,
        "distances": [float(x) for x in distances],
        "used_llm": True,
    }


def _faiss_write(index: faiss.Index, dest: Path) -> None:
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


def ensure_berlin_index(force: bool = False) -> dict:
    from berlin_documents import DOCUMENTS

    name = "berlin_facts"
    if not force:
        loaded = load_index(name)
        if loaded is not None and len(loaded[0]) == len(DOCUMENTS):
            docs, index = loaded
            return {"documents": docs, "index": index, "n": len(docs)}

    docs = list(DOCUMENTS)
    emb = embed_texts(docs, batch_size=16)
    index = build_faiss(emb)
    save_index(name, docs, emb, index)
    return {"documents": docs, "index": index, "n": len(docs)}
