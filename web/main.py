import json
import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
).rstrip("/")
MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"))
SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "Ты локальный ассистент. Отвечай кратко и только по фактам. "
    "Если данных недостаточно — так и скажи, не выдумывай.",
)
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
TOP_K = int(os.getenv("LLM_TOP_K", "40"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))

RAG_ENABLED = os.getenv("RAG_ENABLED", "1").strip() not in {"0", "false", "False"}
RAG_BASE = os.getenv("RAG_BASE", "resumes_no_tags")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_DISTANCE = float(os.getenv("RAG_DISTANCE", "55"))

_rag_pack = None


def _provider_label() -> str:
    if RAG_ENABLED:
        return f"RAG/{RAG_BASE}"
    return "Ollama" if LLM_BACKEND == "ollama" else "llama-server"


def _repo_root() -> Path:
    env = os.getenv("REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _load_rag():
    global _rag_pack
    if _rag_pack is not None:
        return _rag_pack
    root = _repo_root()
    hw = root / "labs" / "ML-HW10"
    if str(hw) not in sys.path:
        sys.path.insert(0, str(hw))
    from rag_system import GenParams, ensure_resume_indices, rag_answer

    packs = ensure_resume_indices(force=False)
    if RAG_BASE not in packs:
        raise KeyError(f"Unknown RAG_BASE={RAG_BASE}, available={list(packs)}")
    _rag_pack = {
        "pack": packs[RAG_BASE],
        "rag_answer": rag_answer,
        "GenParams": GenParams,
    }
    return _rag_pack


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{{MODEL_LABEL}}", MODEL)
    html = html.replace("{{PROVIDER_LABEL}}", _provider_label())
    return html


async def _stream_ollama(message: str):
    payload = {
        "model": MODEL,
        "stream": True,
        "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "num_predict": MAX_TOKENS,
        },
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{LLM_BASE_URL}/api/chat",
            json=payload,
        ) as response:
            if response.status_code != 200:
                error = await response.aread()
                yield f"data: {json.dumps({'error': error.decode()}, ensure_ascii=False)}\n\n"
                return
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = (data.get("message") or {}).get("content") or data.get("response")
                if content:
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                if data.get("done"):
                    break


async def _stream_rag(message: str):
    import anyio

    rag = _load_rag()
    params = rag["GenParams"](
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_new_tokens=MAX_TOKENS,
    )

    def _run():
        return rag["rag_answer"](
            message,
            rag["pack"]["index"],
            rag["pack"]["documents"],
            top_k=RAG_TOP_K,
            distance_threshold=RAG_DISTANCE,
            params=params,
            lang="ru",
            model=MODEL,
        )

    out = await anyio.to_thread.run_sync(_run)
    answer = out.get("answer") or ""
    # отдаём одним SSE-событием (RAG уже полный ответ)
    yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def generate():
        try:
            if RAG_ENABLED:
                async for chunk in _stream_rag(request.message):
                    yield chunk
            else:
                async for chunk in _stream_ollama(request.message):
                    yield chunk
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Ollama недоступен. Запустите ollama serve.'}, ensure_ascii=False)}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'error': str(error)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    info = {
        "status": "ok",
        "provider": _provider_label(),
        "backend": LLM_BACKEND,
        "base_url": LLM_BASE_URL,
        "model": MODEL,
        "rag_enabled": RAG_ENABLED,
        "rag_base": RAG_BASE if RAG_ENABLED else None,
        "local": True,
    }
    if RAG_ENABLED:
        try:
            pack = _load_rag()["pack"]
            info["rag_docs"] = pack["index"].ntotal
        except Exception as exc:
            info["rag_error"] = str(exc)
    return info
