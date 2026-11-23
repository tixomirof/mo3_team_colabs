"""Общее ядро сравнения RAG по обезличенным резюме."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import ZipFile

import httpx
import numpy as np

def _repo_root():
    env = os.getenv("HW11_REPO_DIR")
    if env:
        return Path(env).resolve()
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        if (directory / "data" / "ML-HW11").exists() or (directory / "labs" / "ML-HW10" / "rag_system.py").is_file():
            return directory
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
WORK_DATA = Path(os.getenv("HW11_DATA_DIR", str(ROOT / "data" / "ML-HW11")))
INDEX_BUNDLE = Path(os.getenv("HW11_INDEX_BUNDLE", str(WORK_DATA / "indices.zip")))
MODEL = "qwen3:4b-instruct"
TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ANSWER_LIMIT = 512
CHUNK_SIZE = 256
LONG_CANDIDATES = 100
VECTOR_CHECKPOINT_SIZE = 512
MODES = ("naive", "local", "global", "hybrid", "mix")
REFUSAL = "В резюме нет достаточных сведений для ответа."
INSTRUCTION = (
    "Ответь по-русски и прямо на вопрос, не более четырёх коротких предложений. "
    "Приводи только факты, нужные для ответа. Используй только предоставленные "
    "резюме. Описания графа помогают найти связь, а подтверждение бери из исходных "
    "фрагментов. Ставь ссылку [ID] после каждого утверждения. Различай кандидатов "
    "по ID. Если подтверждена только часть ответа, укажи её и объясни, чего не хватает. "
    f"Если подтвердить ответ невозможно, напиши: «{REFUSAL}»"
)


def runtime_dir():
    return Path(os.getenv("HW11_RUNTIME_DIR", str(Path.home() / ".cache" / "mo3-hw11")))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def citation_ids(answer):
    normalized = answer.translate(str.maketrans({"С": "C", "с": "C", "Р": "R"}))
    return sorted(set(re.findall(r"\[(?:ID\s+)?([CRU]\d+(?:-v\d+)?)\]", normalized)))


@dataclass(frozen=True)
class Resume:
    id: str
    category: str
    text: str
    candidate: int | None = None
    index_id: str | None = None

    @property
    def key(self):
        return self.index_id or self.id

    @property
    def document(self):
        title = f"Кандидат №{self.candidate}" if self.candidate else f"Резюме {self.key}"
        return f"{title}\n{self.text}"


def load_corpus(size, directory=None):
    if size != 400:
        raise ValueError("В сравнении используются 400 резюме.")
    directory = directory or (
        Path.cwd() if (Path.cwd() / "resumes_selected.csv").exists() else ROOT / "data"
    )
    portable = (directory if directory.name == "ML-HW11" else directory / "ML-HW11") / "resumes.csv"
    if portable.is_file():
        with portable.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == ["index", "text", "class", "candidate_id"]:
                rows = []
                for row in reader:
                    position = int(row["candidate_id"])
                    anchor = position if 1 <= position <= 18 else None
                    key = f"C{anchor:02d}" if anchor else f"R{row['index']}"
                    rows.append(Resume(f"R{position}", row["class"], row["text"].replace("\r\n", "\n"),
                                       anchor, key))
            else:
                rows = [
                    Resume(row["id"], row["category"], row["text"].replace("\r\n", "\n"),
                           int(row["candidate"]) if row["candidate"] else None, row["index_id"])
                    for row in reader
                ]
        if ([row.id for row in rows] != [f"R{number}" for number in range(1, size + 1)]
                or len({row.key for row in rows}) != size
                or [row.candidate for row in rows[:18]] != list(range(1, 19))):
            raise ValueError("Корпус ML-HW11 должен содержать R1–R400 в порядке следования.")
        return rows
    with (directory / "resumes_selected.csv").open(encoding="utf-8-sig", newline="") as stream:
        selected = {
            row["index"]: Resume(f"C{int(row['candidate_id']):02d}", row["class"],
                                 row["text"].replace("\r\n", "\n"), int(row["candidate_id"]))
            for row in csv.DictReader(stream)
        }
    if len(selected) != 18 or len({r.id for r in selected.values()}) != 18:
        raise ValueError("Ожидаются 18 резюме с уникальными ID.")
    with (directory / "resumes_ru_6000_labeled.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        full = {row["index"]: row for row in csv.DictReader(stream)}
    if len(full) != 6000 or not selected.keys() <= full.keys():
        raise ValueError("Полный набор должен содержать 6 000 ID, включая прежних кандидатов.")
    for key, resume in selected.items():
        if full[key]["text"].split() != resume.text.split():
            raise ValueError(f"Содержание {resume.id} различается между корпусами.")
    rows = [*selected.values(), *(
        Resume(f"R{key}", row["class"], row["text"].replace("\r\n", "\n"))
        for key, row in full.items() if key not in selected
    )]
    return [replace(row, id=f"R{position}", index_id=row.id)
            for position, row in enumerate(rows[:size], start=1)]


@lru_cache(maxsize=1)
def legacy():
    os.environ.setdefault("RAG_CACHE_DIR", str(runtime_dir() / "legacy"))
    sys.path.insert(0, str(ROOT / "labs" / "ML-HW10"))
    import rag_system

    return rag_system


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.getenv("HW11_QWEN_TOKENIZER", TOKENIZER))


class Tokens:
    def encode(self, text):
        return tokenizer().encode(text, add_special_tokens=False, verbose=False)

    def decode(self, ids):
        return tokenizer().decode(ids, clean_up_tokenization_spaces=False)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, batches):
        return [self.decode(ids) for ids in batches]

    def count(self, text):
        return len(self.encode(text))

    def fragments(self, text, size):
        offsets = tokenizer()(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False,
        )["offset_mapping"]
        for start in range(0, len(offsets), size):
            end = min(start + size, len(offsets))
            yield text[offsets[start][0]:offsets[end - 1][1]]


class IncompleteGeneration(RuntimeError):
    def __init__(self, answer, reason, input_tokens, output_tokens):
        super().__init__("Ollama вернул пустой или незавершённый ответ.")
        self.result = {
            "answer": answer, "reason": reason,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        }


class Ollama:
    def __init__(self, url, context=16384):
        self.url = url.strip().rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Укажите адрес Ollama в OLLAMA_BASE_URL.")
        self.context = context
        self.digest = ""
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

    async def connect(self):
        tags = await self.http.get(f"{self.url}/api/tags")
        tags.raise_for_status()
        model = next((row for row in tags.json()["models"] if row["name"] == MODEL), None)
        if model is None:
            raise RuntimeError(f"В Ollama нет модели {MODEL}.")
        self.digest = model["digest"]
        await self.complete([{"role": "user", "content": "Ответь одним словом: готово."}],
                            temperature=0.1, limit=16)
        active = await self.http.get(f"{self.url}/api/ps")
        active.raise_for_status()
        model = next((row for row in active.json()["models"] if row["name"] == MODEL), {})
        if not model.get("context_length"):
            raise RuntimeError("Ollama не сообщил размер выделенного окна.")
        self.context = min(self.context, model["context_length"])
        return self.context

    async def complete(self, messages, *, temperature, limit, seed=None):
        options = {
            "temperature": temperature, "num_ctx": self.context,
            "num_predict": limit, "top_p": 0.8, "top_k": 50,
        }
        if seed is not None:
            options["seed"] = seed
        response = await self.http.post(f"{self.url}/api/chat", json={
            "model": MODEL, "stream": False, "think": False,
            "messages": messages, "options": options,
        })
        response.raise_for_status()
        data = response.json()
        answer = data["message"]["content"].strip()
        if not answer or not data.get("done") or data.get("done_reason") == "length":
            raise IncompleteGeneration(answer, data.get("done_reason"),
                                       data.get("prompt_eval_count"), data.get("eval_count"))
        return answer, data["prompt_eval_count"]

    async def light_completion(
        self, prompt, system_prompt=None, history_messages=None, *, completion_limit=4096, **kwargs
    ):
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        content, _ = await self.complete(messages, temperature=0.1, limit=completion_limit, seed=42)
        return content


@dataclass(frozen=True)
class Config:
    method: str
    mode: str = "mix"
    group_size: int = 2000
    top_groups: int = 3
    light_top_k: int = 20
    chunk_top_k: int = 10
    budget: int = 12000
    temperature: float = 0.1


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str


@dataclass(frozen=True)
class Block:
    sources: tuple[Evidence, ...]
    note: str = ""
    references: tuple[str, ...] = ()

    @property
    def text(self):
        return "\n".join(filter(None, (self.note, *(
            f"[{source.id}] {source.text}" for source in self.sources
        ))))


class ResumeIndex:
    def __init__(self, resumes, directory, *, vector_bundle=INDEX_BUNDLE):
        self.resumes = resumes
        self.directory = directory
        self.vector_bundle = vector_bundle
        self.id_by_document = {row.document: row.key for row in resumes}
        self.public_ids = {row.key: row.id for row in resumes}
        self.index_ids = {row.id: row.key for row in resumes}
        self.signature = fingerprint([(r.key, r.category, r.document) for r in resumes])
        self.tokens = Tokens()
        self._groups = {}

    def faiss_bundle_folder(self, name):
        folder = "resumes_no_tags" if name == "documents" else f"longrag_{name}"
        return f"faiss/{self.signature}-{fingerprint(EMBEDDINGS)}/{folder}"

    def vectors(self, name, texts):
        path = self.directory / f"{self.signature}-{name}-{fingerprint(EMBEDDINGS)}.npy"
        if path.exists():
            return np.load(path, allow_pickle=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.vector_bundle is not None and self.vector_bundle.is_file():
            with ZipFile(self.vector_bundle) as archive:
                names = set(archive.namelist())
                prepared = f"{self.faiss_bundle_folder(name)}/embeddings.npy"
                member = prepared if prepared in names else f"vectors/{path.name}"
                if member in names:
                    with archive.open(member) as stream:
                        vectors = np.load(stream, allow_pickle=False)
                    if vectors.shape != (len(texts), 384) or vectors.dtype != np.float32:
                        raise ValueError(f"Несовместимый размер или тип векторов для {name}.")
                    self._write_vectors(path, vectors)
                    return vectors
        if len(texts) <= VECTOR_CHECKPOINT_SIZE:
            vectors = legacy().embed_texts(texts)
        else:
            # 512 кратно исходной партии MiniLM из 32 текстов. Состав партий сохраняется.
            folder = path.with_suffix("")
            folder.mkdir(exist_ok=True)
            parts = []
            for start in range(0, len(texts), VECTOR_CHECKPOINT_SIZE):
                part = folder / f"{start:06d}.npy"
                if not part.exists():
                    self._write_vectors(part, legacy().embed_texts(texts[start:start + VECTOR_CHECKPOINT_SIZE]))
                parts.append(np.load(part, allow_pickle=False))
            vectors = np.concatenate(parts)
        self._write_vectors(path, vectors)
        return vectors

    @staticmethod
    def _write_vectors(path, vectors):
        stream = NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                np.save(stream, vectors, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @cached_property
    def baseline_vectors(self):
        return self.vectors("documents", list(self.id_by_document))

    @cached_property
    def baseline_index(self):
        return legacy().build_faiss(self.baseline_vectors)

    def baseline(self, question):
        old = legacy()
        ids = old.extract_candidate_ids(question)
        if ids and not any(row.candidate in ids for row in self.resumes):
            return [], {}
        hits, distances = old.retrieve(question, self.baseline_index, list(self.id_by_document), top_k=3)
        return [
            Block((Evidence(self.id_by_document[doc], doc),))
            for doc, distance in zip(hits, distances) if old.is_relevant(distance, 55)
        ], {}

    @cached_property
    def fragments(self):
        return [
            (number, Evidence(row.key, text))
            for number, row in enumerate(self.resumes)
            for text in self.tokens.fragments(row.document, CHUNK_SIZE)
        ]

    @cached_property
    def fragment_index(self):
        return legacy().build_faiss(self.vectors(
            f"chunks-{CHUNK_SIZE}", [source.text for _, source in self.fragments],
        ))

    def groups(self, max_size):
        if max_size in self._groups:
            return self._groups[max_size]
        part_numbers, categories = defaultdict(list), defaultdict(list)
        for part, (number, _) in enumerate(self.fragments):
            part_numbers[number].append(part)
        groups, by_fragment = {}, {}

        def store(sources, parts):
            group_id = f"G{len(groups) + 1:04d}"
            groups[group_id] = Block(tuple(sources), f"Группа {group_id}")
            by_fragment.update(dict.fromkeys(parts, group_id))

        evidence = [Evidence(row.key, row.document) for row in self.resumes]
        lengths = [self.tokens.count(Block((item,)).text) for item in evidence]
        for number, row in enumerate(self.resumes):
            if lengths[number] + 12 <= max_size:
                categories[row.category].append(number)
                continue
            # Части длинного документа сохраняют его исходный ID.
            parts, sources = [], []
            for part in part_numbers[number]:
                item = self.fragments[part][1]
                if parts and self.tokens.count(Block((*sources, item)).text) + 12 > max_size:
                    store(sources, parts)
                    parts, sources = [], []
                parts.append(part)
                sources.append(item)
            store(sources, parts)

        # В категории к первому ещё свободному резюме подбираются ближайшие соседи.
        for category in categories.values():
            remaining = set(category)
            while remaining:
                seed = min(remaining)
                vector = self.baseline_vectors[seed]
                nearest = sorted(
                    remaining - {seed},
                    key=lambda n: (np.square(vector - self.baseline_vectors[n]).sum(), n),
                )
                numbers = [seed]
                count = lengths[seed] + 12
                for number in nearest:
                    if count + lengths[number] + 2 <= max_size:
                        numbers.append(number)
                        count += lengths[number] + 2
                remaining.difference_update(numbers)
                store([evidence[n] for n in numbers], [p for n in numbers for p in part_numbers[n]])
        self._groups[max_size] = groups, by_fragment
        return groups, by_fragment

    def long(self, question, config):
        groups, by_fragment = self.groups(config.group_size)
        vector = legacy().embed_texts([legacy().search_text(question)])
        distances, positions = self.fragment_index.search(
            vector, min(LONG_CANDIDATES, len(self.fragments)),
        )
        # Первое попадание в отсортированной выдаче FAISS даёт минимум L2 группы.
        best = {}
        for distance, position in zip(distances[0], positions[0]):
            group = by_fragment[position]
            item = self.fragments[position][1]
            best.setdefault(group, {"distance": float(distance), "fragment": item.text, "source_id": item.id})
        order = list(best)[:config.top_groups]
        return [groups[group] for group in order], {"groups": {g: best[g] for g in order}}


def messages(question, blocks, system_prompt=INSTRUCTION):
    context = "\n\n".join(block.text for block in blocks) or "Нет найденных материалов."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Материалы:\n{context}\n\nВопрос: {question}"},
    ]


def pack(question, blocks, budget, system_prompt=INSTRUCTION):
    accepted = []
    sources = set()
    for block in blocks:
        if not set(block.references) <= sources:
            continue
        chat = messages(question, [*accepted, block], system_prompt)
        prompt = tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if Tokens().count(prompt) + 32 <= budget:
            accepted.append(block)
            sources.update(item.id for item in block.sources)
    return accepted


class Study:
    def __init__(self, index, ollama, light=None, *, system_prompt=INSTRUCTION):
        self.index = index
        self.ollama = ollama
        self.light = light
        self.system_prompt = system_prompt

    def budget(self, requested):
        ceiling = min(12000, self.ollama.context - ANSWER_LIMIT)
        if requested == 12000:
            return ceiling
        return 8000 if ceiling > 8000 else ceiling * 2 // 3

    async def retrieve(self, question, config):
        start = perf_counter()
        budget = self.budget(config.budget)
        index_question = re.sub(r"\bR\d+\b", lambda match: self.index.index_ids.get(
            match.group(), match.group()), question)
        if config.method == "baseline":
            blocks, trace = await asyncio.to_thread(self.index.baseline, index_question)
        elif config.method == "long":
            blocks, trace = await asyncio.to_thread(self.index.long, index_question, config)
        else:
            if self.light is None:
                raise RuntimeError("Индекс LightRAG ещё не открыт.")
            blocks, trace = await self.light.retrieve(index_question, config, budget)
        blocks, trace = self.public_context(blocks, trace)
        return pack(question, blocks, budget, self.system_prompt), trace, perf_counter() - start

    def public_context(self, blocks, trace):
        # Привязка к позиции меняет подписи, сохраняя текст документов и построенный поиск.
        public = self.index.public_ids

        def label(value):
            return public.get(value, value)

        def display_text(text):
            return re.sub(r"\b[CR]\d+\b", lambda match: label(match.group()), text)

        visible = [Block(
            tuple(Evidence(label(source.id), display_text(source.text)) for source in block.sources),
            display_text(block.note),
            tuple(label(key) for key in block.references),
        ) for block in blocks]
        if "groups" in trace:
            trace = {**trace, "groups": {
                key: {**match, "source_id": label(match["source_id"]), "fragment": display_text(match["fragment"])}
                for key, match in trace["groups"].items()
            }}
        if "edges" in trace:
            trace = {**trace, "edges": [
                {**edge, "from": display_text(edge["from"]), "to": display_text(edge["to"]),
                 "description": display_text(edge["description"]),
                 "sources": [label(key) for key in edge["sources"]]}
                for edge in trace["edges"]
            ]}
        return visible, trace

    async def answer(self, question, config, context=None, seed=42):
        blocks, trace, search_seconds = (
            await self.retrieve(question, config) if context is None else context
        )
        start = perf_counter()
        if not blocks:
            answer, actual = REFUSAL, 0
        else:
            answer, actual = await self.ollama.complete(
                messages(question, blocks, self.system_prompt),
                temperature=config.temperature, limit=ANSWER_LIMIT, seed=seed,
            )
        if actual > self.budget(config.budget):
            raise RuntimeError("Фактический вход Ollama превысил общий бюджет.")
        sources = defaultdict(list)
        for block in blocks:
            for item in block.sources:
                if item.text not in sources[item.id]:
                    sources[item.id].append(item.text)
        group_ids = {block.note.removeprefix("Группа ") for block in blocks if block.note.startswith("Группа ")}
        if "groups" in trace:
            trace = {"groups": {key: val for key, val in trace["groups"].items() if key in group_ids}}
        if "edges" in trace:
            trace = {**trace, "edges": [
                edge for edge in trace["edges"] if set(edge["sources"]) <= sources.keys()
            ]}
        return {
            "answer": answer, "input_tokens": actual,
            "seconds": search_seconds + perf_counter() - start, "retrieval_seconds": search_seconds,
            "sources": [{"id": key, "text": "\n".join(parts)} for key, parts in sources.items()],
            "used_ollama": bool(blocks),
            "cited": citation_ids(answer),
            "trace": trace, "effective_budget": self.budget(config.budget),
        }
