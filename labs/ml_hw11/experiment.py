"""Воспроизводимый порядок запусков и ручной оценки."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .core import (
    ANSWER_LIMIT, EMBEDDINGS, Block, Config, Evidence, IncompleteGeneration,
    MODES, Resume, ResumeIndex, citation_ids, fingerprint, legacy,
)
from .questions import Question, questions_for

BASELINE = Config("baseline")
LONG = Config("long")
LIGHT = Config("light")
INITIAL = (BASELINE, LONG, *(replace(LIGHT, mode=mode) for mode in MODES))
VARIANTS = (
    replace(LONG, group_size=1000),
    replace(LONG, top_groups=5),
    replace(LONG, budget=8000),
    replace(LIGHT, light_top_k=5),
    replace(LIGHT, chunk_top_k=3),
    replace(LIGHT, budget=8000),
)
TEMPERATURES = (0.1, 0.4, 0.7)
TEMPERATURE_REPETITIONS = 4
REVIEWS_FILENAME = "оценки.json"
VERSION = 2

UPDATE_OLD = Resume(
    "U017-v1", "Учебный пример",
    "Вымышленное резюме. Алекс Маяков (ID U017) разработал модуль проверки данных "
    "проекта «Контроль качества». Сведений о работе с января 2026 года нет.",
)
UPDATE_OTHER = Resume(
    "U208-v1", "Учебный пример",
    "Вымышленное резюме. Алекс Маяков (ID U208) использовал Power BI для банковской отчётности.",
)
UPDATE_NEW = Resume(
    "U017-v2", "Учебный пример",
    "Вымышленное резюме. Алекс Маяков (ID U017) разработал модуль проверки данных проекта "
    "«Контроль качества». С января 2026 года тестирует учебную систему «Маяк».",
)
UPDATE_BEFORE = (
    Question("U1", "Новый факт", "Над какой системой работает Алекс Маяков (ID U017) с января 2026 года?",
             "В резюме U017 нет сведений о работе с января 2026 года.", ()),
    Question("U2", "Прежний факт", "В каком проекте Алекс Маяков (ID U017) разработал модуль проверки данных?",
             "Проект «Контроль качества».", ("U017-v1",)),
    Question("U3", "Различение объектов", "Подтверждено ли, что Алекс Маяков (ID U017) использовал Power BI?",
             "Нет сведений о Power BI у U017.", ()),
)
UPDATE_AFTER = (
    replace(UPDATE_BEFORE[0], expected="С января 2026 года тестирует систему «Маяк».", sources=("U017-v2",)),
    replace(UPDATE_BEFORE[1], sources=("U017-v2",)),
    UPDATE_BEFORE[2],
)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def label(config):
    if config.method == "baseline":
        return "Базовая RAG"
    if config.method == "long":
        return "LongRAG"
    return f"LightRAG {config.mode}"


def context_distances(index, questions):
    """Квадраты L2 до полных резюме для общей диагностики контекста."""
    retrieval = legacy()
    documents = index.baseline_vectors
    public_ids = [index.public_ids[key] for key in index.id_by_document.values()]
    distances = {}
    for question in questions:
        vector = retrieval.embed_texts([retrieval.search_text(question.text)])[0]
        squared_l2 = ((documents - vector) ** 2).sum(axis=1)
        distances[question.id] = dict(zip(public_ids, map(float, squared_l2), strict=True))
    return distances


class Experiment:
    def __init__(self, study, size, *, section="comparison"):
        self.study = study
        self.size = size
        self.questions = questions_for(size)
        index_key = fingerprint([
            study.index.signature, study.index.public_ids, study.ollama.digest, study.ollama.context,
        ])
        self.index_report = study.index.directory / "index-timing" / f"{index_key}.json"
        self.path = study.index.directory / "runs" / fingerprint([
            VERSION, index_key, study.system_prompt, ANSWER_LIMIT, [asdict(question) for question in self.questions],
            *(() if section == "comparison" else (section,)),
        ])
        self.selected_config = self.path / "selected.json"
        self.reviews = self.path / REVIEWS_FILENAME

    def _key(self, phase, question, config, repetition):
        return fingerprint([self.path.name, phase, asdict(question), asdict(config), repetition])

    async def run_one(self, phase, question, config, repetition=0, context=None):
        key = self._key(phase, question, config, repetition)
        output = self.path / phase / f"{key}.json"
        existing = self.cached_row(phase, question, config, repetition)
        if existing is not None:
            return {**existing, "cited": citation_ids(existing["answer"])}
        for attempt in (1, 2):
            try:
                result = await self.study.answer(question.text, config, context, seed=42 + repetition)
            except IncompleteGeneration as error:
                failure = {"key": key, "question": question.id, "attempt": attempt, **error.result}
                save_json(self.path / "errors" / f"{key}-{attempt}.json", failure)
                if attempt == 2:
                    raise
            except Exception as error:
                # Сообщения HTTP клиента могут содержать адрес туннеля.
                failure = {"key": key, "question": question.id, "type": type(error).__name__}
                save_json(self.path / "errors" / f"{key}-technical.json", failure)
                raise
            else:
                break
        result.update({
            "key": key, "set": self.size, "phase": phase, "question": asdict(question),
            "config": asdict(config), "method": label(config), "repetition": repetition,
            "cached_context": context is not None, "cached_answer": False, "attempt": attempt,
        })
        save_json(output, result)
        return result

    async def run(self, phase, configs, questions=None, progress=None):
        rows = []
        for config in configs:
            for question in self.questions if questions is None else questions:
                row = await self.run_one(phase, question, config)
                rows.append(row)
                if progress:
                    progress(row)
        return rows

    async def temperature_context(self, question, base):
        key = self._key("temperature-context", question, base, 0)
        path = self.path / "temperature-context" / f"{key}.json"
        saved = read_json(path)
        if saved is None:
            blocks, trace, _ = await self.study.retrieve(question.text, base)
            saved = {"blocks": [asdict(block) for block in blocks], "trace": trace}
            save_json(path, saved)
        blocks = [
            Block(tuple(Evidence(**item) for item in block["sources"]),
                  block["note"], tuple(block["references"]))
            for block in saved["blocks"]
        ]
        # При возобновлении используется прежний контекст. Замер охватывает генерацию.
        return blocks, saved["trace"], 0.0

    async def temperature(self, configs, progress=None):
        rows = []
        for base in configs:
            for question in self.questions:
                context = await self.temperature_context(question, base)
                for temperature in TEMPERATURES:
                    for repetition in range(1, TEMPERATURE_REPETITIONS + 1):
                        row = await self.run_one(
                            "temperature", question, replace(base, temperature=temperature),
                            repetition, context,
                        )
                        rows.append(row)
                        if progress:
                            progress(row)
        return rows

    def rows(self, phase):
        return [read_json(path) for path in sorted((self.path / phase).glob("*.json"))]

    def cached_rows(self, phase, config, questions):
        return [
            row for question in questions
            if (row := self.cached_row(phase, question, config)) is not None
        ]

    def cached_row(self, phase, question, config, repetition=0):
        key = self._key(phase, question, config, repetition)
        return read_json(self.path / phase / f"{key}.json")

    def review(self, row, score, found, comment=""):
        target = set(row["question"]["sources"])
        observed = {source["id"] for source in row["sources"]}
        found = sorted(set(found))
        if score not in (0, 1, 2) or len(found) > len(target) or not set(found) <= observed:
            raise ValueError("Оценка должна быть 0–2. Учитывайте подтверждающие резюме из переданного контекста.")
        scores = read_json(self.reviews, {})
        scores[row["key"]] = {"score": score, "found": found, "comment": comment}
        save_json(self.reviews, scores)

    def with_reviews(self, rows):
        reviews = read_json(self.reviews, {})
        return [{**row, **reviews.get(row["key"], {"score": None, "found": None, "comment": ""})}
                for row in rows]

    def select(self, initial, variants):
        selected = []
        question_ids = {question.id for question in self.questions}
        reviews = self.with_reviews([
            row for row in initial + variants
            if row["config"]["budget"] == 12000
        ])
        for method in ("long", "light"):
            candidates = {}
            for row in reviews:
                config = Config(**row["config"])
                if config.method == method:
                    candidates.setdefault(config, []).append(row)
            if not candidates or any(
                len(rows) != len(question_ids)
                or {row["question"]["id"] for row in rows} != question_ids
                or any(row["score"] is None for row in rows)
                for rows in candidates.values()
            ):
                return None

            def rank(config):
                rows = candidates[config]
                return (
                    -sum(row["score"] for row in rows),
                    -sum(len(row["found"]) for row in rows),
                    sum(row["seconds"] for row in rows),
                )

            selected.append(min(candidates, key=rank))
        return tuple(selected)

    def select_temperatures(self, configs, rows):
        selected = []
        expected = {(question.id, repetition) for question in self.questions
                    for repetition in range(1, TEMPERATURE_REPETITIONS + 1)}
        evaluated = self.with_reviews(rows)
        for base in configs:
            candidates = []
            for temperature in TEMPERATURES:
                config = replace(base, temperature=temperature)
                runs = [row for row in evaluated if row["config"] == asdict(config)]
                if len(runs) != len(expected) or {
                    (row["question"]["id"], row["repetition"]) for row in runs
                } != expected or any(row["score"] is None for row in runs):
                    return None
                rank = (-sum(row["score"] for row in runs),
                        -sum(row["score"] > 0 for row in runs),
                        sum(row["seconds"] for row in runs))
                candidates.append((rank, config))
            selected.append(min(candidates, key=lambda candidate: candidate[0])[1])
        return tuple(selected)


async def build_local(index, group_sizes):
    rows = []
    for method, attr in (("Базовая RAG", "baseline_index"), ("LongRAG", "fragment_index")):
        start = perf_counter()
        built = await asyncio.to_thread(getattr, index, attr)
        rows.append({"метод": method, "единиц": built.ntotal, "секунды": perf_counter() - start})
    for size in group_sizes:
        start = perf_counter()
        groups, _ = await asyncio.to_thread(index.groups, size)
        rows.append({"метод": f"Группировка LongRAG ({size})", "единиц": len(groups),
                     "секунды": perf_counter() - start})
    return rows


async def measure_local_indices(index, group_sizes):
    key = fingerprint([index.signature, EMBEDDINGS, group_sizes])
    path = index.directory / "index-timing" / f"faiss-cold-{key}.json"
    report = read_json(path)
    if report is None:
        with TemporaryDirectory(dir=index.directory) as directory:
            cold_index = ResumeIndex(index.resumes, Path(directory), vector_bundle=None)
            report = await build_local(cold_index, group_sizes)
        save_json(path, report)
    return report


def export_config(path: Path, configs):
    save_json(path, {config.method: asdict(config) for config in configs})
