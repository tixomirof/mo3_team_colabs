"""LightRAG HKUDS с единым эмбеддером и источниками доказательств."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj, copytree
from tempfile import TemporaryDirectory
from time import perf_counter, time_ns
from uuid import uuid4
from zipfile import ZipFile

from .core import (
    Block, EMBEDDINGS, Evidence, INDEX_BUNDLE, IncompleteGeneration, MODEL, TOKENIZER, Tokens,
    fingerprint, legacy,
)
from .experiment import read_json, save_json

VERSION = "1.5.7"
CHUNK_TOKENS = 480
CHUNK_OVERLAP = 48
SNAPSHOT_FILES = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json", "kv_store_entity_chunks.json", "kv_store_full_docs.json",
    "kv_store_full_entities.json", "kv_store_full_relations.json", "kv_store_relation_chunks.json",
    "kv_store_text_chunks.json", "vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json",
)


@dataclass(frozen=True)
class IndexReport:
    documents: int
    seconds: float
    reused: int


class LightIndex:
    def __init__(self, index, ollama, *, variant="main", seed_directory=None):
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc, Tokenizer

        signature = fingerprint([
            index.signature, ollama.digest, ollama.context, VERSION,
            TOKENIZER, EMBEDDINGS, CHUNK_TOKENS, CHUNK_OVERLAP, variant, 2,
        ])
        directory = index.directory / f"light-{signature}"
        directory.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.timing = directory / "index-build.json"
        # Идентификатор workspace изолирует хранилища в одном процессе ноутбука.
        workspace = "" if variant == "main" and len(index.resumes) == 18 else f"hw11-{signature}"
        self.directory = directory / workspace
        if seed_directory and not self.directory.exists():
            with TemporaryDirectory(dir=directory) as temporary:
                seed = Path(temporary) / "seed"
                copytree(seed_directory, seed)
                seed.replace(self.directory)
        elif variant == "main" and not self.directory.exists() and INDEX_BUNDLE.is_file():
            # Имя workspace совпадает только при том же корпусе, моделях и размере окна.
            with ZipFile(INDEX_BUNDLE) as archive:
                prefix = f"graph/{workspace}/"
                names = {prefix + name for name in SNAPSHOT_FILES}
                if names <= set(archive.namelist()):
                    with TemporaryDirectory(dir=directory) as temporary:
                        snapshot = Path(temporary) / "index"
                        snapshot.mkdir()
                        for name in SNAPSHOT_FILES:
                            with archive.open(prefix + name) as src, (snapshot / name).open("wb") as dst:
                                copyfileobj(src, dst)
                        snapshot.replace(self.directory)

        embed_lock = asyncio.Lock()

        async def embed(texts):
            async with embed_lock:
                return await asyncio.to_thread(legacy().embed_texts, texts)

        async def complete(prompt, **kwargs):
            limit = 4096
            for attempt in (1, 2):
                try:
                    return await ollama.light_completion(prompt, completion_limit=limit, **kwargs)
                except IncompleteGeneration as error:
                    save_json(
                        directory / "incomplete" / f"{time_ns()}-{fingerprint(prompt)}.json",
                        {"attempt": attempt, "limit": limit, **error.result},
                    )
                    if attempt == 2:
                        raise
                    if error.result["reason"] == "length":
                        available = ollama.context - (error.result["input_tokens"] or 0) - 128
                        limit = max(limit, min(8192, available))

        self.rag = LightRAG(
            working_dir=str(directory),
            workspace=workspace,
            tokenizer=Tokenizer(TOKENIZER, Tokens()),
            llm_model_name=MODEL,
            llm_model_func=complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=384, max_token_size=512, func=embed,
            ),
            chunk_token_size=CHUNK_TOKENS,
            chunk_overlap_token_size=CHUNK_OVERLAP,
            addon_params={"language": "Russian"},
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=True,
            max_parallel_insert=4,
            llm_model_max_async=4,
        )

    async def __aenter__(self):
        await self.rag.initialize_storages()
        return self

    async def __aexit__(self, *exc):
        await self.rag.finalize_storages()

    async def processed(self, ids):
        from lightrag.base import DocStatus

        states = await self.rag.aget_docs_by_ids(ids)
        return {key for key, row in states.items() if row["status"] == DocStatus.PROCESSED}

    async def add(self, documents):
        if documents:
            await self.rag.ainsert(
                [row.document for row in documents],
                ids=[row.key for row in documents],
                file_paths=[row.key for row in documents],
            )
        done = await self.processed([row.key for row in documents]) if documents else set()
        missing = {row.key for row in documents} - done
        if missing:
            raise RuntimeError(f"LightRAG не завершил индексирование {len(missing)} резюме.")

    async def resume_failed(self):
        from lightrag.base import DocStatus
        from lightrag.kg.shared_storage import (
            commit_manual_retry_request, get_namespace_data, get_namespace_lock, get_pipeline_ingress,
        )

        failed = await self.rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        expected = {row.key for row in self.index.resumes} & failed.keys()
        if not expected:
            return False
        # В LightRAG 1.5.7 документ со статусом FAILED возобновляется
        # по тому же явному запросу, который использует его серверный API.
        workspace = self.rag.workspace
        status = await get_namespace_data("pipeline_status", workspace=workspace)
        lock = get_namespace_lock("pipeline_status", workspace=workspace)
        ingress = await get_pipeline_ingress(workspace)
        refusal = await commit_manual_retry_request(status, lock, ingress, uuid4().hex, {})
        if refusal:
            raise RuntimeError(f"LightRAG не разрешил возобновить индексацию: {refusal}")
        await self.rag.apipeline_process_enqueue_documents()
        if expected - await self.processed(list(expected)):
            raise RuntimeError("После возобновления остались необработанные резюме.")
        return True

    async def build(self, *, batch_size=48, progress=None):
        from lightrag.base import DocStatus

        rows = self.index.resumes
        ids = {row.key for row in rows}
        start = perf_counter()
        states = await self.rag.doc_status.get_docs_by_statuses([DocStatus.PROCESSED])
        initial = ids & states.keys()
        has_work = len(initial) < len(rows)
        previous = read_json(self.timing, {}).get("seconds", 0.0)
        done = len(initial)
        try:
            if has_work and await self.resume_failed():
                states = await self.rag.doc_status.get_docs_by_statuses([DocStatus.PROCESSED])
                done = len(ids & states.keys())
            remaining = [row for row in rows if row.key not in states]
            for offset in range(0, len(remaining), batch_size):
                batch = remaining[offset:offset + batch_size]
                await self.add(batch)
                done += len(batch)
                elapsed = previous + perf_counter() - start
                save_json(self.timing, {"documents": done, "seconds": elapsed})
                if progress:
                    progress(done, len(rows), elapsed)
        finally:
            seconds = previous + perf_counter() - start if has_work else previous
            if has_work:
                save_json(self.timing, {"documents": done, "seconds": seconds})
        return IndexReport(len(rows), seconds, len(initial))

    async def replace(self, old_id, document):
        previous = await self.rag.aget_docs_by_ids([old_id])
        if old_id in previous:
            result = await self.rag.adelete_by_doc_id(old_id, delete_llm_cache=True)
            if result.status != "success":
                raise RuntimeError(f"Не удалось удалить прежнюю версию {old_id}.")
        if not await self.processed([document.id]):
            await self.add([document])

    async def retrieve(self, question, config, budget):
        from lightrag import QueryParam
        from lightrag.constants import GRAPH_FIELD_SEP

        result = await self.rag.aquery_data(question, QueryParam(
            mode=config.mode,
            only_need_context=True,
            enable_rerank=False,
            top_k=config.light_top_k,
            chunk_top_k=config.chunk_top_k,
            max_entity_tokens=3000,
            max_relation_tokens=3000,
            max_total_tokens=budget,
            include_references=True,
        ))
        data = result.get("data") or {}
        chunks = data.get("chunks", [])
        relationships = data.get("relationships", [])
        entities = data.get("entities", [])
        by_chunk = {row["chunk_id"]: Evidence(row["file_path"], row["content"]) for row in chunks}
        blocks = [Block((item,)) for item in by_chunk.values()]
        requested = set(re.findall(r"\bID\s+([CRU]\d+(?:-v\d+)?)\b", question, re.I))
        if requested:
            targets = {item.casefold() for item in requested}
            # Одно имя в графе может соединять опыт двух кандидатов.
            # При явном ID доказательства берём из резюме именно этого объекта.
            scoped = [
                block for block in blocks
                if block.sources[0].id.casefold() in targets
                or block.sources[0].id.casefold().split("-v", 1)[0] in targets
            ]
            return scoped, {"edges": [], "chunk_count": len(scoped)}
        edges = []
        graph_blocks = []

        def references(row):
            keys = row.get("source_id", "").split(GRAPH_FIELD_SEP)
            return list(dict.fromkeys(by_chunk[key].id for key in keys if key in by_chunk))

        for row in relationships:
            sources = references(row)
            if sources:
                edge = {
                    "from": row["src_id"], "to": row["tgt_id"],
                    "description": row["description"], "sources": sources,
                }
                edges.append(edge)
                note = (f"Связь {edge['from']} → {edge['to']}: {edge['description']} "
                        + " ".join(f"[{key}]" for key in sources))
                graph_blocks.append(Block((), note, tuple(sources)))
        for row in entities:
            sources = references(row)
            if sources:
                note = (f"Сущность {row['entity_name']}: {row['description']} "
                        + " ".join(f"[{key}]" for key in sources))
                graph_blocks.append(Block((), note, tuple(sources)))
        return blocks + graph_blocks, {"edges": edges, "entity_count": len(entities), "chunk_count": len(chunks)}
