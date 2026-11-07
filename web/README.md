# Local AI Chat + RAG по резюме

Чат в браузере. По умолчанию отвечает **RAG** по `data/resumes_ru_6000_labeled.csv` через Ollama (`qwen3:4b-instruct`).

## Запуск

1. Ollama с моделью:
   ```bash
   ollama pull qwen3:4b-instruct
   ```
2. Один раз построить FAISS (кэш в `~/.cache/mo3_ml_hw10_rag/`):
   ```bash
   cd labs/ML-HW10
   set LLM_MODEL=qwen3:4b-instruct
   python -c "from rag_system import ensure_resume_indices; print({k:v['index'].ntotal for k,v in ensure_resume_indices().items()})"
   ```
3. Web:
   ```bash
   cd web
   set LLM_BACKEND=ollama
   set LLM_MODEL=qwen3:4b-instruct
   set RAG_ENABLED=1
   set RAG_BASE=resumes_no_tags
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
4. Открыть http://127.0.0.1:8000

## Переменные

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `RAG_ENABLED` | `1` | RAG вкл/выкл |
| `RAG_BASE` | `resumes_no_tags` | или `resumes_with_tags` |
| `RAG_TOP_K` | `3` | сколько документов в контекст |
| `RAG_DISTANCE` | `55` | порог L2; выше = жёстче «нет данных» |
| `LLM_MODEL` | `qwen3:4b-instruct` | модель Ollama |

Ноутбук сдачи: `labs/ML-HW10/ML-HW10_RAG_resumes.ipynb`.
