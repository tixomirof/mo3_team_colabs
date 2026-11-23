"""ML-HW11: тегированная база («Категория: X») против нетегированной.

Команды (из корня репозитория, нужен OLLAMA_BASE_URL):
  python -m labs.ml_hw11.tags_compare build     # FAISS базы и фрагменты LongRAG с тегами
  python -m labs.ml_hw11.tags_compare inject    # быстрый граф LightRAG: копия + теги без LLM
  python -m labs.ml_hw11.tags_compare run untagged|tagged
  python -m labs.ml_hw11.tags_compare rebuild   # честная пересборка графа с тегами (часы)

Все артефакты с тегами лежат в runtime_dir()/tags и не пересекаются с рабочими индексами.
Тег добавляется так же, как в ML-HW10: «Кандидат №N\nКатегория: X\n<текст>».
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(os.getenv("HW11_REPO_DIR", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))

from labs.ml_hw11.core import (  # noqa: E402
    INSTRUCTION, Config, IncompleteGeneration, Ollama, ResumeIndex, Study, load_corpus, runtime_dir,
)
from labs.ml_hw11.experiment import read_json, save_json  # noqa: E402
from labs.ml_hw11.light import LightIndex  # noqa: E402
from labs.ml_hw11.questions import questions_for  # noqa: E402

TAGS_DIR = runtime_dir() / "tags"
OUT = Path(os.getenv("HW11_TAGS_OUT", str(TAGS_DIR / "results")))
# Итоговые настройки ноутбука (ячейка 25, selected.json): seed 42, общий INSTRUCTION.
FINAL = {
    "baseline": Config("baseline"),
    "long": Config("long", group_size=2000, top_groups=5, temperature=0.1),
    "light": Config("light", mode="mix", light_top_k=5, chunk_top_k=10, temperature=0.7),
}
CATEGORY_TYPE = "category"
CANDIDATE_TYPE = "candidate"


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def with_tag(row):
    return replace(row, text=f"Категория: {row.category}\n{row.text}")


def make_index(base):
    rows = load_corpus(400)
    if base == "untagged":
        return ResumeIndex(rows, runtime_dir())
    return ResumeIndex([with_tag(row) for row in rows], TAGS_DIR, vector_bundle=None)


async def connect():
    ollama = Ollama(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    context=int(os.getenv("HW11_OLLAMA_CTX", "16384")))
    await ollama.connect()
    return ollama


def build():
    index = make_index("tagged")
    report = {}
    for name, action in (
        ("faiss_documents", lambda: index.baseline_index),
        ("longrag_fragments_256", lambda: index.fragment_index),
        ("longrag_groups_2000", lambda: index.groups(2000)),
    ):
        start = time.perf_counter()
        action()
        report[name] = round(time.perf_counter() - start, 2)
        log(name, report[name], "s")
    report["fragments"] = len(index.fragments)
    report["groups_2000"] = len(index.groups(2000)[0])
    report["signature"] = index.signature
    save_json(OUT / "build_tagged.json", report)
    log(report)


def untagged_graph_dir(ollama):
    index = make_index("untagged")
    light = LightIndex(index, ollama)
    return light.directory


async def inject():
    """Теги в готовый граф без LLM: узел категории, узел резюме и ребро резюме → категория."""
    from lightrag.constants import GRAPH_FIELD_SEP
    from lightrag.utils import compute_mdhash_id, make_relation_chunk_key

    ollama = await connect()
    seed = untagged_graph_dir(ollama)
    if not (seed / "graph_chunk_entity_relation.graphml").is_file():
        raise RuntimeError(f"Нет исходного графа {seed}")
    index = make_index("tagged")
    start = time.perf_counter()
    light = LightIndex(index, ollama, variant="tags-inject", seed_directory=seed)
    marker = light.directory.parent / "tags-injected.json"
    if marker.is_file():
        log("Теги уже внесены:", marker)
        return
    log("seed", seed, "->", light.directory, round(time.perf_counter() - start, 1), "s")
    rows = {row.key: row for row in index.resumes}
    chunks = read_json(light.directory / "kv_store_text_chunks.json")
    first = {}
    for chunk_id, chunk in chunks.items():
        if chunk["chunk_order_index"] == 0:
            first[chunk["full_doc_id"]] = chunk_id
    if set(first) != set(rows):
        raise RuntimeError(f"Нет первого фрагмента для {len(set(rows) - set(first))} резюме.")
    now = int(time.time())
    stats = {"chunks": 0, "candidates": 0, "candidate_nodes_existing": 0, "edges": 0, "categories": {}}
    async with light:
        rag = light.rag
        graph = rag.chunk_entity_relation_graph

        def title(row):
            return f"Кандидат №{row.candidate}" if row.candidate else f"Резюме {row.key}"

        # 1. Строка категории в текст: первый фрагмент, полный документ, вектор фрагмента.
        text_updates, vdb_updates, doc_updates = {}, {}, {}
        for key, row in rows.items():
            line = f"Категория: {row.category}"
            chunk = await rag.text_chunks.get_by_id(first[key])
            head, _, rest = chunk["content"].partition("\n")
            if not head.startswith(("Кандидат №", "Резюме ")):
                raise RuntimeError(f"Неожиданное начало фрагмента {first[key]}: {head[:40]}")
            content = f"{head}\n{line}\n{rest}"
            chunk.pop("_id", None)
            text_updates[first[key]] = {
                **chunk, "content": content, "tokens": light.rag.tokenizer.encode(content).__len__(),
                "update_time": now,
            }
            vdb_updates[first[key]] = {"content": content, "full_doc_id": key, "file_path": chunk["file_path"]}
            doc = await rag.full_docs.get_by_id(key)
            doc.pop("_id", None)
            h, _, r = doc["content"].partition("\n")
            doc_updates[key] = {**doc, "content": f"{h}\n{line}\n{r}", "update_time": now}
        await rag.text_chunks.upsert(text_updates)
        await rag.chunks_vdb.upsert(vdb_updates)
        await rag.full_docs.upsert(doc_updates)
        stats["chunks"] = len(text_updates)

        # 2. Узлы категорий (один на категорию, источники — первые фрагменты её резюме).
        by_category = {}
        for key, row in rows.items():
            by_category.setdefault(row.category, []).append(key)
        entity_vdb, entity_chunks = {}, {}
        for category, keys in sorted(by_category.items()):
            if await graph.has_node(category):
                raise RuntimeError(f"В графе уже есть узел «{category}»")
            ids = [first[key] for key in keys]
            description = (f"Категория резюме «{category}» из разметки датасета. "
                           f"К категории отнесено резюме: {len(keys)}.")
            node = {
                "entity_id": category, "entity_type": CATEGORY_TYPE, "description": description,
                "source_id": GRAPH_FIELD_SEP.join(ids), "file_path": GRAPH_FIELD_SEP.join(keys),
                "created_at": now, "truncate": "",
            }
            await graph.upsert_node(category, node)
            entity_chunks[category] = {"chunk_ids": ids, "count": len(ids)}
            entity_vdb[compute_mdhash_id(category, prefix="ent-")] = {
                "content": f"{category}\n{description}", "entity_name": category,
                "source_id": node["source_id"], "description": description,
                "entity_type": CATEGORY_TYPE, "file_path": node["file_path"],
            }
            stats["categories"][category] = len(keys)

        # 3. Узел резюме и ребро резюме → категория.
        relation_vdb, relation_chunks = {}, {}
        full_entities, full_relations = {}, {}
        for key, row in rows.items():
            name, category, chunk_id = title(row), row.category, first[key]
            if await graph.has_node(name):
                stats["candidate_nodes_existing"] += 1
                existing = await graph.get_node(name)
                ids = existing["source_id"].split(GRAPH_FIELD_SEP)
                if chunk_id not in ids:
                    raise RuntimeError(f"Узел {name} относится к другому резюме")
            else:
                description = f"{name}: резюме кандидата из категории «{category}»."
                await graph.upsert_node(name, {
                    "entity_id": name, "entity_type": CANDIDATE_TYPE, "description": description,
                    "source_id": chunk_id, "file_path": key, "created_at": now, "truncate": "",
                })
                entity_chunks[name] = {"chunk_ids": [chunk_id], "count": 1}
                entity_vdb[compute_mdhash_id(name, prefix="ent-")] = {
                    "content": f"{name}\n{description}", "entity_name": name, "source_id": chunk_id,
                    "description": description, "entity_type": CANDIDATE_TYPE, "file_path": key,
                }
                stats["candidates"] += 1
            keywords = "категория,сфера деятельности"
            description = f"{name} относится к категории «{category}»."
            await graph.upsert_edge(name, category, {
                "weight": 1.0, "description": description, "keywords": keywords,
                "source_id": chunk_id, "file_path": key, "created_at": now, "truncate": "",
            })
            src, tgt = sorted((name, category))
            relation_vdb[compute_mdhash_id(src + tgt, prefix="rel-")] = {
                "src_id": src, "tgt_id": tgt, "source_id": chunk_id,
                "content": f"{keywords}\t{src}\n{tgt}\n{description}", "keywords": keywords,
                "description": description, "weight": 1.0, "file_path": key,
            }
            relation_chunks[make_relation_chunk_key(name, category)] = {"chunk_ids": [chunk_id], "count": 1}
            entities = await rag.full_entities.get_by_id(key) or {"entity_names": [], "count": 0}
            entities.pop("_id", None)
            names = list(dict.fromkeys([*entities["entity_names"], name, category]))
            full_entities[key] = {**entities, "entity_names": names, "count": len(names), "update_time": now}
            relations = await rag.full_relations.get_by_id(key) or {"relation_pairs": [], "count": 0}
            relations.pop("_id", None)
            pairs = [*relations["relation_pairs"], [src, tgt]]
            full_relations[key] = {**relations, "relation_pairs": pairs, "count": len(pairs), "update_time": now}
            stats["edges"] += 1
        await rag.entities_vdb.upsert(entity_vdb)
        await rag.relationships_vdb.upsert(relation_vdb)
        await rag.entity_chunks.upsert(entity_chunks)
        await rag.relation_chunks.upsert(relation_chunks)
        await rag.full_entities.upsert(full_entities)
        await rag.full_relations.upsert(full_relations)
        await rag._insert_done()
    stats["seconds"] = round(time.perf_counter() - start, 2)
    stats["directory"] = str(light.directory)
    save_json(marker, stats)
    save_json(OUT / "inject_tagged_graph.json", stats)
    log(stats)
    await ollama.http.aclose()


async def run(base):
    ollama = await connect()
    index = make_index(base)
    kwargs = {} if base == "untagged" else {"variant": "tags-inject"}
    light = LightIndex(index, ollama, **kwargs)
    if base == "tagged" and not (light.directory.parent / "tags-injected.json").is_file():
        raise RuntimeError("Сначала выполните inject.")
    log(base, "graph:", light.directory)
    path = OUT / f"answers_{base}.json"
    rows = read_json(path, [])
    done = {(row["method"], row["question_id"]) for row in rows}
    async with light:
        study = Study(index, ollama, light, system_prompt=INSTRUCTION)
        for method, config in FINAL.items():
            for question in questions_for(400):
                if (method, question.id) in done:
                    continue
                error = None
                for attempt in (1, 2):
                    try:
                        result = await study.answer(question.text, config, seed=42)
                        break
                    except IncompleteGeneration as failure:
                        error = failure.result
                        result = None
                retrieved = [] if result is None else [item["id"] for item in result["sources"]]
                expected = list(question.sources)
                row = {
                    "base": base, "method": method, "config": asdict(config),
                    "question_id": question.id, "kind": question.kind, "question": question.text,
                    "expected": question.expected, "expected_sources": expected,
                    "answer": None if result is None else result["answer"],
                    "cited": [] if result is None else result["cited"],
                    "retrieved": retrieved,
                    "retrieval_hit": (None if not expected else sum(s in retrieved for s in expected) / len(expected)),
                    "seconds": None if result is None else round(result["seconds"], 2),
                    "retrieval_seconds": None if result is None else round(result["retrieval_seconds"], 2),
                    "input_tokens": None if result is None else result["input_tokens"],
                    "error": error,
                }
                rows.append(row)
                save_json(path, rows)
                log(base, method, question.id, row["seconds"], "hit", row["retrieval_hit"],
                    (row["answer"] or "ERROR")[:100].replace("\n", " "))
    await ollama.http.aclose()


async def rebuild():
    """Честная пересборка графа LightRAG по тегированным документам (умеренная нагрузка на Ollama)."""
    import lightrag

    parallel = int(os.getenv("HW11_TAGS_PARALLEL", "2"))
    original = lightrag.LightRAG

    def limited(*args, **kwargs):
        kwargs["max_parallel_insert"] = parallel
        kwargs["llm_model_max_async"] = parallel
        return original(*args, **kwargs)

    ollama = await connect()
    index = make_index("tagged")
    lightrag.LightRAG = limited
    try:
        light = LightIndex(index, ollama, variant="tags-full")
    finally:
        lightrag.LightRAG = original
    log("rebuild dir:", light.directory, "parallel", parallel)
    async with light:
        report = await light.build(batch_size=8, progress=lambda done, count, seconds:
                                   log(f"LightRAG tags-full: {done}/{count}, {seconds:.0f} s"))
    save_json(OUT / "rebuild_tagged_graph.json", {**asdict(report), "directory": str(light.directory)})
    log("done", report)
    await ollama.http.aclose()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "build":
        build()
    elif command == "inject":
        asyncio.run(inject())
    elif command == "run":
        asyncio.run(run(sys.argv[2]))
    elif command == "rebuild":
        asyncio.run(rebuild())
    else:
        raise SystemExit(__doc__)
