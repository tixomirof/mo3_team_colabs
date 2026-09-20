import json
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с чат-консолью"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Проксирует поток генерации от локального Ollama в SSE."""
    async def generate():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": MODEL, "prompt": request.message, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        yield f"data: {json.dumps({'error': error.decode()})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("response"):
                            yield f"data: {json.dumps({'content': data['response']})}\n\n"
                        if data.get("done"):
                            break
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Ollama недоступен. Запустите Docker Compose и скачайте модель.'})}\n\n"
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Модель отвечает слишком долго.'})}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'error': str(error)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "provider": "Ollama",
        "model": MODEL,
        "local": True,
    }
