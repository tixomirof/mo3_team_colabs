# Local AI Chat + RAG по резюме

Чат в браузере. Одна модель Ollama (`qwen3:4b-instruct`) и три подхода поиска: базовая RAG из ML-HW10, LongRAG и LightRAG.

## Запуск / остановка (web + LLM вместе)

```powershell
cd web
.\start.ps1          # поднимает Ollama на хосте и контейнер chat_web
.\stop.ps1           # гасит контейнер и Ollama/llama-server
```

Открыть http://127.0.0.1:8000

В шапке чата выберите подход. Для LightRAG появится режим (`naive`, `local`, `global`, `hybrid`, `mix`; по умолчанию `mix`). Все три подхода отвечают одним коротким итоговым текстом по общему промпту. Под ответом — свёрнутый список «Источники (N)» только с теми резюме, на которые ответ ссылается ([R123]); у LongRAG добавляется group_id, у LightRAG — краткие связи. Если ссылок нет, блок не показывается.

`start.ps1` / `stop.ps1` — основной способ: LLM живёт вместе с вебом.

Перед запуском положите готовый архив индексов в `data/ML-HW11/indices.zip` и корпус в `data/ML-HW11/resumes.csv` (как в репозитории команды). Другой путь к архиву: переменная `HW11_INDEX_BUNDLE` (внутри контейнера файл должен быть виден, проще оставить его в репозитории). Граф LightRAG при старте **не пересчитывается**.

Проверка: http://127.0.0.1:8000/health — поля `hw11.ready` и `hw11.light_ready`. Для демо удобны вопросы F1 («химическая завивка») и N2 (Kubernetes — ожидается отказ).

## Переменные

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `RAG_ENABLED` | `1` | RAG вкл/выкл |
| `RAG_ENGINE` | `auto` | `auto` — ML-HW11, если индекс поднялся; иначе прежний RAG 10-й лабы. `legacy` принудительно включает старый путь |
| `RAG_BASE` | `resumes_no_tags` | индекс 10-й лабы, если HW11 не загрузился |
| `RAG_TOP_K` | `5` | сколько документов в контекст (только legacy) |
| `RAG_DISTANCE` | `55` | порог L2 (только legacy) |
| `LLM_MODEL` | `qwen3:4b-instruct` | модель Ollama, общая для трёх подходов |
| `HW11_INDEX_BUNDLE` | `/repo/data/ML-HW11/indices.zip` | путь к готовому архиву индексов |
| `HW11_OLLAMA_CTX` | `16384` | окно Ollama; снимок LightRAG собран с этим значением. На RTX 3060 Ti 8 ГБ держите лишние процессы закрытыми |
| `HW11_MOCK_LLM` | unset | `1` — заглушка без Ollama, только для проверки API |

Итоговые настройки поиска из ноутбука: LongRAG — группы до 2000 токенов, 5 групп, T=0.1; LightRAG mix — `top_k=5`, `chunk_top_k=10`, T=0.7; бюджет входа 12 000.

## Один раз: FAISS-кэш 10-й лабы (fallback)

```powershell
cd labs\ML-HW10
$env:LLM_MODEL="qwen3:4b-instruct"
python -c "from rag_system import ensure_resume_indices; print({k:v['index'].ntotal for k,v in ensure_resume_indices().items()})"
```

Ноутбуки: `labs/Lab05_ML-HW10_resumes.ipynb`, `labs/Lab06_ML-HW11_LongRAG_LightRAG_resumes.ipynb`.
