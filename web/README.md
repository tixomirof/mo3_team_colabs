# Local AI Chat + RAG по резюме

Чат в браузере. RAG + Ollama (`qwen3:4b-instruct`).

## Запуск / остановка (web + LLM вместе)

```powershell
cd web
.\start.ps1          # поднимает Ollama на хосте и контейнер chat_web
.\stop.ps1           # гасит контейнер и Ollama/llama-server
```

Открыть http://127.0.0.1:8000

`start.ps1` / `stop.ps1` — основной способ: LLM живёт вместе с вебом (старт/стоп одной командой).

## Переменные

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `RAG_ENABLED` | `1` | RAG вкл/выкл |
| `RAG_BASE` | `resumes_no_tags` | или `resumes_with_tags` |
| `RAG_TOP_K` | `5` | сколько документов в контекст |
| `RAG_DISTANCE` | `55` | порог L2 |
| `LLM_MODEL` | `qwen3:4b-instruct` | модель Ollama |

## Один раз: FAISS-кэш

```powershell
cd labs\ML-HW10
$env:LLM_MODEL="qwen3:4b-instruct"
python -c "from rag_system import ensure_resume_indices; print({k:v['index'].ntotal for k,v in ensure_resume_indices().items()})"
```

Ноутбук сдачи: `labs/ML-HW10/ML-HW10_RAG_resumes.ipynb`.
