"""Обёртка ML-HW11 для прежнего веб-чата: три подхода и показ источников."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

APPROACHES = ("baseline", "long", "light")
MODES = ("naive", "local", "global", "hybrid", "mix")
DEFAULT_MODE = "mix"
# Итоговые настройки поиска из ноутбука Lab06.
# Температура чата чуть выше 0.1, чтобы модель не сжимала ответ до одной ссылки.
SELECTED = {
    "baseline": {"method": "baseline", "temperature": 0.25, "budget": 12000},
    "long": {
        "method": "long", "group_size": 2000, "top_groups": 5,
        "temperature": 0.1, "budget": 12000,
    },
    "light": {
        "method": "light", "mode": "mix", "light_top_k": 5, "chunk_top_k": 10,
        "temperature": 0.25, "budget": 12000,
    },
}

REFUSAL = "В резюме нет достаточных сведений для ответа."
CHAT_INSTRUCTION = (
    "Ответь по-русски кратко: не больше четырёх коротких предложений или компактного списка. "
    "Ссылки только вида [R123] (латинская R, в квадратных скобках) и только на реальные ID "
    "из переданных материалов. Не пиши «Р123», «Р[123]», «**R123**» и не выводи литерал «[ID]». "
    "Если вопрос просит одного лучшего («кто больше всего подходит»), назови одного кандидата "
    "и кратко почему, со ссылкой [R123]. "
    "Если вопрос просит нескольких людей, две разные роли, сравнение, перечисление или обобщение "
    "(«какие», «кто из кандидатов», «кого из двух», «сравни»), перечисли каждого подходящего: "
    "у каждого своя ссылка [Rxx] и одна короткая причина. Не своди такой ответ к одному "
    "«наиболее подходящему», если в материалах есть несколько релевантных резюме. "
    "Если подтверждено только одно резюме, достаточно его. "
    "Откажись, только если в материалах нет ничего релевантного вопросу: тогда напиши "
    f"ровно одну фразу «{REFUSAL}». "
    "Если есть хотя бы частично подходящие резюме, ответь по ним и не отказывайся. "
    "Не перечисляй нерелевантных кандидатов и не пиши рассуждения. "
    "Не отвечай одной только ссылкой без предложения."
)
CHAT_TEMPERATURE = 0.25
SNIPPET_LIMIT = 150
MAX_RELATIONS = 3
CITATION_RE = re.compile(
    r"\[(?:ID\s+)?([CRU]\d+(?:-v\d+)?)\]|\b([CRU]\d+(?:-v\d+)?)\b",
    re.IGNORECASE,
)


def repo_root() -> Path:
    env = os.getenv("HW11_REPO_DIR") or os.getenv("REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for directory in (here.parent, *here.parents):
        if (directory / "labs" / "ml_hw11").is_dir() or (directory / "ml_hw11").is_dir():
            if (directory / "data" / "ML-HW11").exists() or (directory / "labs" / "ML-HW10").is_dir() or (
                directory / "ML-HW10"
            ).is_dir():
                return directory
        if (directory / "labs" / "ml_hw11").is_dir() and (directory / "labs" / "ML-HW10").is_dir():
            return directory
    return here.parents[1]


def _prepare_imports(root: Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not (root / "labs" / "ml_hw11").is_dir() and (root / "ml_hw11").is_dir():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def corpus_available(root: Path | None = None) -> bool:
    root = root or repo_root()
    portable = root / "data" / "ML-HW11" / "resumes.csv"
    data_env = os.getenv("HW11_DATA_DIR")
    if data_env and (Path(data_env) / "resumes.csv").is_file():
        return True
    return portable.is_file()


def bundle_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    env = os.getenv("HW11_INDEX_BUNDLE")
    if env:
        return Path(env)
    data = Path(os.getenv("HW11_DATA_DIR", str(root / "data" / "ML-HW11")))
    return data / "indices.zip"


def chat_model() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen3:4b-instruct"


def approach_label(approach: str, mode: str | None = None) -> str:
    if approach == "baseline":
        return "Базовая RAG"
    if approach == "long":
        return "LongRAG"
    return f"LightRAG {mode or 'mix'}"


def _drop_leading_partial_word(text: str) -> str:
    """Если фрагмент начат с середины слова, отбрасываем обрубок до первого пробела."""
    text = text or ""
    if not text:
        return text
    first, _, rest = text.partition(" ")
    if rest and first and first[0].islower() and len(first) <= 2:
        return rest.lstrip()
    return text


def _preview(text: str, limit: int = SNIPPET_LIMIT) -> str:
    collapsed = _drop_leading_partial_word(" ".join((text or "").split()))
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _normalize_id_text(text: str) -> str:
    return (text or "").translate(str.maketrans({"С": "C", "с": "C", "Р": "R", "р": "R"}))


_ID_LETTER = r"[CRUСсРр]"


def _latin_letter(raw: str) -> str:
    char = (raw or "R")[0]
    return {"С": "C", "с": "C", "Р": "R", "р": "R", "У": "U", "у": "U"}.get(char, char.upper())


def normalize_citations(text: str) -> str:
    """Сводит «Р141», «Р[338]», «**R353**», «[Р141]» и голое R353 к каноническому [R353]."""
    text = text or ""

    def repl(letter: str, number: str, suffix: str = "") -> str:
        return f"[{_latin_letter(letter)}{int(number)}{suffix}]"

    text = re.sub(
        rf"\*\*({_ID_LETTER})\s*(\d+)\*\*",
        lambda m: repl(m.group(1), m.group(2)),
        text,
    )
    text = re.sub(
        rf"({_ID_LETTER})\s*\[(\d+)\]",
        lambda m: repl(m.group(1), m.group(2)),
        text,
    )
    text = re.sub(
        rf"\[(?:ID\s+)?({_ID_LETTER})\s*(\d+)((?:-v\d+)?)\]",
        lambda m: repl(m.group(1), m.group(2), m.group(3)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(?<!\[)({_ID_LETTER})(\d+)(?!\])",
        lambda m: repl(m.group(1), m.group(2)),
        text,
    )
    return text


def fix_citation_periods(text: str) -> str:
    """Убирает двойную точку вида «головы. [R1].» → «головы [R1].»."""
    return re.sub(
        r"([.!?…])\s*(\[[CRU]\d+(?:-v\d+)?\])\s*[.!?…]?",
        r" \2.",
        text or "",
        flags=re.IGNORECASE,
    )


def cited_ids(answer: str) -> list[str]:
    """ID вида [R123], R123, Р141, Р[338], **R353** и «кандидат №N» в порядке появления."""
    found = []
    seen = set()

    def add(label: str) -> None:
        if label and label not in seen:
            seen.add(label)
            found.append(label)

    normalized = normalize_citations(answer or "")
    for match in CITATION_RE.finditer(_normalize_id_text(normalized)):
        add((match.group(1) or match.group(2) or "").upper())
    for number in named_candidate_numbers(answer or ""):
        add(f"R{number}")
    return found


def _strip_answer_prefix(text: str) -> str:
    text = (text or "").strip()
    marker = re.search(r"(?is)\bОтвет:\s*", text)
    if marker:
        text = text[marker.end():].strip()
    text = re.sub(r"\[ID\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]{2,}", " ", text).strip(" \n\t-—;:,.")
    return text


def _core_text(text: str) -> str:
    # Не нормализуем кириллицу во всём тексте: иначе «резюме» превращается в «Rезюме».
    cleaned = re.sub(r"\[(?:ID\s+)?[CRUСсРр]\d+(?:-v\d+)?\]", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\[ID\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[CRU]\d+(?:-v\d+)?\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"[\s\[\]().,;:!?«»\"\-—–]+", " ", cleaned).strip()


def is_full_refusal(text: str) -> bool:
    """Схлопывать в REFUSAL только если весь ответ — отказ, без другой фразы."""
    core = re.sub(r"\s+", " ", _core_text(text)).strip(" .!")
    return bool(core) and core.casefold() == REFUSAL.rstrip(".").casefold()


def is_citation_only(text: str) -> bool:
    return bool(cited_ids(text)) and not _core_text(text)


def ensure_period(text: str) -> str:
    """Ставит точку в конце, в том числе после завершающей ссылки [R123]."""
    text = (text or "").rstrip()
    if not text or text.endswith((".", "!", "?", "…")):
        return text
    return text + "."


def polish_answer(text: str) -> str:
    """Обрезка префикса «Ответ:» и литерала [ID]; отказ схлопывается только целиком."""
    text = _strip_answer_prefix(text)
    if is_full_refusal(text):
        return REFUSAL
    text = normalize_citations(text)
    text = fix_citation_periods(text)
    return ensure_period(text)


def expand_citation_only(text: str) -> str:
    """Если модель вернула только ссылку, собираем одно полное предложение."""
    labels = cited_ids(text)
    if not labels:
        return text
    if len(labels) == 1:
        return f"Наиболее подходящий кандидат — [{labels[0]}]."
    cited = ", ".join(f"[{label}]" for label in labels)
    return f"Наиболее подходящие кандидаты — {cited}."


def _lookup(retrieved: dict[str, dict], label: str) -> dict | None:
    if label in retrieved:
        return retrieved[label]
    wanted = label.casefold()
    for key, item in retrieved.items():
        if key.casefold() == wanted:
            return item
    return None



def _resume_position(resume) -> int | None:
    label = str(getattr(resume, "id", "") or "")
    if len(label) > 1 and label[0] in "Rr" and label[1:].isdigit():
        return int(label[1:])
    candidate = getattr(resume, "candidate", None)
    return int(candidate) if candidate else None


def _id_number(label: str | None, prefix: str | None = None) -> int | None:
    text = str(label or "").strip()
    match = re.match(r"^([CRUСРСусу])\s*0*(\d+)$", text, re.IGNORECASE)
    if not match:
        return None
    if prefix and match.group(1).upper().replace("С", "C").replace("У", "U") not in prefix:
        return None
    return int(match.group(2))


def resume_aliases(resume) -> list[str]:
    """Публичный id (R{N}) и C-алиасы того же номера. index_id корпуса 6000 не смешиваем."""
    aliases = []
    public = str(getattr(resume, "id", "") or "")
    if public:
        aliases.append(public)
    candidate = getattr(resume, "candidate", None)
    number = int(candidate) if candidate else _id_number(public, "R")
    if number:
        for alias in (f"C{number}", f"C{number:02d}", f"C{number:03d}"):
            if alias not in aliases:
                aliases.append(alias)
    return aliases


def _same_resume(public_id: str, cited: str) -> bool:
    if not public_id or not cited:
        return False
    if public_id.casefold() == cited.casefold():
        return True
    cited_number = _id_number(cited, "C")
    public_number = _id_number(public_id, "R")
    return cited_number is not None and cited_number == public_number


def raw_resume_dirs(root: Path) -> list[Path]:
    data = Path(os.getenv("HW11_DATA_DIR", str(root / "data" / "ML-HW11")))
    seen = []
    for path in (
        data / "resumes_selected_raw",
        root / "data" / "ML-HW11" / "resumes_selected_raw",
        root / "data" / "resumes_selected_raw",
    ):
        resolved = path
        if resolved not in seen:
            seen.append(resolved)
    return seen


def read_raw_resume(root: Path, position: int) -> str | None:
    """Ищет candidate_NNN_*.txt (HW11) или candidate_NN_*.txt (HW10), без пользовательского пути."""
    if not position or position < 1:
        return None
    patterns = [f"candidate_{position:03d}_*.txt"]
    if position < 100:
        patterns.append(f"candidate_{position:02d}_*.txt")
    for directory in raw_resume_dirs(root):
        if not directory.is_dir():
            continue
        try:
            base = directory.resolve()
        except OSError:
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved.parent != base or not resolved.is_file() or resolved.suffix.lower() != ".txt":
                    continue
                return resolved.read_text(encoding="utf-8")
    return None


def public_ids_map(resumes) -> dict:
    """Как в Study.public_context: ключ корпуса 6000 / C01 → публичный R{N}."""
    mapping = {}
    for row in resumes:
        public = str(getattr(row, "id", "") or "")
        if not public:
            continue
        for key in (getattr(row, "key", None), getattr(row, "index_id", None)):
            if key:
                mapping[str(key)] = public
    return mapping


def public_document(resume, public_ids: dict | None = None) -> str:
    text = getattr(resume, "document", None) or getattr(resume, "text", "") or ""
    if not public_ids:
        return text
    return re.sub(r"\b[CR]\d+\b", lambda match: public_ids.get(match.group(), match.group()), text)


def corpus_texts(resumes, root: Path | None = None) -> dict:
    """Словарь публичных ID (R{N}) и C-алиасов того же номера к полному тексту резюме."""
    table = {}
    public_ids = public_ids_map(resumes)
    for row in resumes:
        position = _resume_position(row)
        full = public_document(row, public_ids)
        if not full and root and position:
            full = read_raw_resume(root, position) or ""
        entry = {
            "id": getattr(row, "id", ""),
            "category": getattr(row, "category", "") or "",
            "full_text": full,
        }
        for alias in resume_aliases(row):
            existing = table.get(alias) or table.get(alias.upper()) or table.get(alias.casefold())
            if existing and existing.get("id") != entry["id"]:
                continue
            table[alias] = entry
            table[alias.upper()] = entry
            table[alias.casefold()] = entry
    return table


def mock_corpus_texts() -> dict:
    r1 = (
        "Кандидат №1\nКатегория: Медицина и здравоохранение\n\n"
        "Расчесывание, стрижка волос взрослых и детей.\n"
        "Завивка волос химическим и электрическим способом (перманент).\n"
        "Окраска волос в различные цвета и оттенки, их обесцвечивание.\n"
    )
    r2 = (
        "Кандидат №2\nКатегория: Медицина и здравоохранение\n\n"
        "Работа с персоналом, закупка товаров оптом, подбор жилья и составление договоров.\n"
        "Работа с клиентами по организационным вопросам.\n"
    )
    r15 = (
        "Кандидат №15\nКатегория: Транспорт и логистика\n\n"
        "Организация перевозок и работа с транспортными документами.\n"
    )
    r367 = (
        "Кандидат №367\nКатегория: Торговля и продажи\n\n"
        "Приготовление блюд, кулинарная обработка продуктов и работа по технологическим картам.\n"
        "Соблюдение санитарных норм на кухне.\n"
    )
    records = [
        type("R", (), {"id": "R1", "key": "C01", "index_id": "C01", "candidate": 1,
                       "category": "Медицина и здравоохранение", "document": r1, "text": r1})(),
        type("R", (), {"id": "R2", "key": "C02", "index_id": "C02", "candidate": 2,
                       "category": "Медицина и здравоохранение", "document": r2, "text": r2})(),
        type("R", (), {"id": "R15", "key": "C15", "index_id": "C15", "candidate": 15,
                       "category": "Транспорт и логистика", "document": r15, "text": r15})(),
        type("R", (), {"id": "R367", "key": "R367", "index_id": "R367", "candidate": None,
                       "category": "Торговля и продажи", "document": r367, "text": r367})(),
    ]
    r18 = (
        "Кандидат №18\nКатегория: Финансы и бухгалтерия\n\n"
        "Организовывал строительные подряды, (малоэтажное строительство деревянных домов).\n"
    )
    r19 = (
        "Резюме R19\nКатегория: Другое\n\n"
        "Совсем другое резюме из корпуса 6000, его нельзя показывать вместо кандидата №1.\n"
    )
    r36 = (
        "Резюме R36\nКатегория: Другое\n\n"
        "Совсем другое резюме, его нельзя показывать вместо кандидата №18.\n"
    )
    records.extend([
        type("R", (), {"id": "R18", "key": "C18", "index_id": "C18", "candidate": 18,
                       "category": "Финансы и бухгалтерия", "document": r18, "text": r18})(),
        type("R", (), {"id": "R19", "key": "R1", "index_id": "R1", "candidate": None,
                       "category": "Другое", "document": r19, "text": r19})(),
        type("R", (), {"id": "R36", "key": "R18", "index_id": "R18", "candidate": None,
                       "category": "Другое", "document": r36, "text": r36})(),
    ])
    return corpus_texts(records, root=None)


def mock_resumes():
    """Те же заглушки корпуса, что и в mock_corpus_texts, списком объектов."""
    table = mock_corpus_texts()
    seen = []
    rows = []
    for entry in table.values():
        public = entry["id"]
        if public in seen:
            continue
        seen.append(public)
        text = entry["full_text"]
        candidate = _id_number(public, "R") if public in {"R1", "R2", "R15", "R18"} else None
        index_id = {"R1": "C01", "R2": "C02", "R15": "C15", "R18": "C18", "R19": "R1", "R36": "R18"}.get(public, public)
        rows.append(type("R", (), {
            "id": public, "key": index_id, "index_id": index_id, "candidate": candidate,
            "category": entry.get("category") or "", "document": text, "text": text,
        })())
    return rows


def present_sources(result: dict, approach: str, mode: str | None, answer: str | None = None, corpus: dict | None = None) -> dict:
    """Только записи, на которые ссылается итоговый ответ, с короткими сниппетами."""
    text = answer if answer is not None else result.get("answer") or ""
    retrieved = {item.get("id", ""): item for item in result.get("sources") or [] if item.get("id")}
    used_ids = []
    for label in cited_ids(text):
        record = _lookup(corpus or {}, label)
        public = record["id"] if record and _same_resume(record.get("id") or "", label) else label
        if _lookup(retrieved, public) or _lookup(retrieved, label) or record:
            if public not in used_ids:
                used_ids.append(public)
        elif public not in used_ids:
            used_ids.append(public)
    payload = {
        "approach": approach,
        "label": approach_label(approach, mode),
        "model": chat_model(),
        "mode": mode if approach == "light" else None,
        "used": bool(used_ids),
        "items": [],
    }
    if not used_ids:
        return payload

    groups = (result.get("trace") or {}).get("groups") or {}
    edges = (result.get("trace") or {}).get("edges") or []
    for label in used_ids:
        item = _lookup(retrieved, label) or _lookup(retrieved, label.upper()) or {}
        record_preview = _lookup(corpus or {}, label) or {}
        entry = {
            "id": item.get("id") or (record_preview.get("id") if record_preview else None) or label,
            "snippet": _preview(item.get("text") or record_preview.get("full_text") or "", SNIPPET_LIMIT),
        }
        record = None
        for key in (label, entry["id"]):
            found = _lookup(corpus or {}, key)
            if found and _same_resume(found.get("id") or "", key):
                record = found
                break
        if record:
            entry["id"] = record["id"]
            if record.get("category"):
                entry["category"] = record["category"]
            entry["full_text"] = record.get("full_text") or item.get("text") or ""
        else:
            entry["full_text"] = item.get("text") or ""
        if approach == "long":
            group_id = next(
                (gid for gid, match in groups.items()
                 if str(match.get("source_id") or "").casefold() == entry["id"].casefold()),
                None,
            )
            if group_id:
                entry["group_id"] = group_id
                fragment = _preview((groups[group_id] or {}).get("fragment") or "", SNIPPET_LIMIT)
                if fragment and not entry["snippet"]:
                    entry["snippet"] = fragment
        if approach == "light":
            relations = []
            for edge in edges:
                sources = [str(value) for value in (edge.get("sources") or [])]
                if not any(value.casefold() == entry["id"].casefold() for value in sources):
                    continue
                left = str(edge.get("from") or "").strip()
                right = str(edge.get("to") or "").strip()
                label_edge = " → ".join(part for part in (left, right) if part)
                if label_edge and label_edge not in relations:
                    relations.append(label_edge)
                if len(relations) >= MAX_RELATIONS:
                    break
            if relations:
                entry["relations"] = relations
        payload["items"].append(entry)
    return payload



_ORDINAL_STEMS = (
    ("девятнадцат", 19), ("восемнадцат", 18), ("семнадцат", 17), ("шестнадцат", 16),
    ("пятнадцат", 15), ("четырнадцат", 14), ("тринадцат", 13), ("двенадцат", 12),
    ("одиннадцат", 11), ("двадцат", 20), ("десят", 10), ("девят", 9), ("восьм", 8),
    ("седьм", 7), ("шест", 6), ("пят", 5), ("четв[её]рт", 4), ("трет(?:ь)?", 3),
    ("втор", 2), ("перв", 1),
)
_ORDINAL_RE = re.compile(
    r"\b(?:" + "|".join(stem for stem, _ in _ORDINAL_STEMS) + r")(?:ый|ой|ий|ая|ое|ого|ому|ым|ом|ую|ей|ья|ью)?\b",
    re.IGNORECASE,
)
_CANDIDATE_ID_PATTERNS = (
    r"кандидат(?:а|у|ом|е)?\s*№?\s*(\d+)",
    r"№\s*(\d+)",
    r"\bno\.?\s*(\d+)\b",
    r"\bid\s*[:=]?\s*(\d+)\b",
    r"\b[RРр]\s*(\d+)\b",
)


def named_candidate_numbers(question: str) -> list[int]:
    """Номера из «кандидат №N», «№N», «R12» и, если есть слово «кандидат», из «первый»."""
    text = question or ""
    numbers: list[int] = []

    def add(value: int) -> None:
        if value > 0 and value not in numbers:
            numbers.append(value)

    for pattern in _CANDIDATE_ID_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add(int(match.group(1)))
    if re.search(r"кандидат", text, flags=re.IGNORECASE):
        for match in _ORDINAL_RE.finditer(text):
            token = match.group(0).casefold()
            for stem, value in _ORDINAL_STEMS:
                if re.match(stem, token, flags=re.IGNORECASE):
                    add(value)
                    break
    return numbers


def resume_by_candidate_number(resumes, number: int):
    """Канонический публичный id — R{N}: сначала поле candidate, иначе id R{N}."""
    if not number:
        return None
    for row in resumes:
        if getattr(row, "candidate", None) == number:
            return row
    wanted = f"R{number}"
    for row in resumes:
        if str(getattr(row, "id", "")).casefold() == wanted.casefold():
            return row
    return None


def resumes_named_in_question(question: str, resumes) -> list:
    found = []
    seen = set()
    for number in named_candidate_numbers(question):
        row = resume_by_candidate_number(resumes, number)
        if row is None:
            continue
        public = getattr(row, "id", "")
        if public in seen:
            continue
        seen.add(public)
        found.append(row)
    return found


def inject_named_into_context(study, question, config, context):
    """В чате добавляет в контекст резюме, явно названные в вопросе (как baseline ID-поиск)."""
    blocks, trace, seconds = context
    resumes = getattr(getattr(study, "index", None), "resumes", None) or []
    wanted = resumes_named_in_question(question, resumes)
    if not wanted:
        return context
    try:
        core = __import__(study.__class__.__module__, fromlist=["Block", "Evidence", "pack"])
    except Exception:
        return context
    present = {item.id for block in blocks for item in getattr(block, "sources", ())}
    extra = []
    for row in wanted:
        public = row.id
        if public in present or getattr(row, "key", public) in present:
            continue
        visible, _ = study.public_context(
            [core.Block((core.Evidence(row.key, row.document),))],
            {},
        )
        extra.extend(visible)
        present.add(public)
    if not extra:
        return context
    budget = study.budget(config.budget)
    packed = core.pack(question, extra + list(blocks), budget, study.system_prompt)
    return packed, trace, seconds


class Hw11Engine:
    def __init__(self, root: Path | None = None):
        self.root = (root or repo_root()).resolve()
        os.environ.setdefault("HW11_REPO_DIR", str(self.root))
        _prepare_imports(self.root)
        self.study = None
        self.light = None
        self.ollama = None
        self.index = None
        self.ready = False
        self.light_ready = False
        self.status = "not-started"
        self.detail = ""
        self.resume_texts = {}

    def config_for(self, approach: str, mode: str = DEFAULT_MODE):
        _prepare_imports(self.root)
        try:
            from labs.ml_hw11.core import Config
        except ImportError:
            from ml_hw11.core import Config
        if approach not in APPROACHES:
            raise ValueError(f"Неизвестный подход: {approach}")
        values = dict(SELECTED[approach])
        values.setdefault("temperature", CHAT_TEMPERATURE)
        if approach == "light":
            if mode not in MODES:
                raise ValueError(f"Неизвестный режим LightRAG: {mode}")
            values["mode"] = mode
        return Config(**values)

    async def start(self) -> None:
        _prepare_imports(self.root)
        try:
            from labs.ml_hw11.core import Ollama, ResumeIndex, Study, load_corpus, runtime_dir
            from labs.ml_hw11.light import LightIndex
        except ImportError:
            from ml_hw11.core import Ollama, ResumeIndex, Study, load_corpus, runtime_dir
            from ml_hw11.light import LightIndex

        bundle = bundle_path(self.root)
        if not corpus_available(self.root):
            self.status = "no-corpus"
            self.detail = (
                "Не найден корпус data/ML-HW11/resumes.csv. "
                "Укажите HW11_REPO_DIR или HW11_DATA_DIR."
            )
            return
        self.index = ResumeIndex(load_corpus(400), runtime_dir())
        self.resume_texts = corpus_texts(self.index.resumes, self.root)
        url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434"
        )
        context = int(os.getenv("HW11_OLLAMA_CTX", os.getenv("OLLAMA_NUM_CTX", "16384")))
        self.ollama = Ollama(url, context=context)
        await self.ollama.connect()
        # Итоговый ответ во всех трёх подходах генерируется здесь по общему чат-промпту.
        # LightRAG по-прежнему отдаёт только контекст (only_need_context=True в модуле лабы).
        self.study = Study(self.index, self.ollama, system_prompt=CHAT_INSTRUCTION)
        self.ready = True
        self.status = "ready"
        self.detail = f"Окно Ollama: {self.ollama.context} токенов."
        if not bundle.is_file():
            self.detail += f" Архив индексов не найден: {bundle}."
        try:
            self.light = LightIndex(self.index, self.ollama)
            await self.light.__aenter__()
            keys = [row.key for row in self.index.resumes]
            processed = await self.light.processed(keys)
            if len(processed) < len(keys):
                await self.light.__aexit__(None, None, None)
                self.light = None
                self.light_ready = False
                self.detail += (
                    " Снимок LightRAG не загрузился или не совпал с моделью и окном Ollama "
                    f"({len(processed)} из {len(keys)} резюме). "
                    "Проверьте data/ML-HW11/indices.zip и HW11_OLLAMA_CTX=16384."
                )
            else:
                self.study.light = self.light
                self.light_ready = True
        except Exception as error:
            self.light = None
            self.light_ready = False
            self.detail += f" LightRAG недоступен: {error}"

    async def close(self) -> None:
        if self.light is not None:
            try:
                await self.light.__aexit__(None, None, None)
            except Exception:
                pass
            self.light = None
        if self.ollama is not None:
            try:
                await self.ollama.http.aclose()
            except Exception:
                pass
            self.ollama = None
        self.ready = False
        self.light_ready = False

    async def answer(self, question: str, approach: str, mode: str = DEFAULT_MODE) -> dict:
        if not self.ready or self.study is None:
            raise RuntimeError(self.detail or "Движок ML-HW11 ещё не готов.")
        if approach == "light" and not self.light_ready:
            raise RuntimeError(self.detail or "Индекс LightRAG не загружен.")
        config = self.config_for(approach, mode)
        # Study.answer не кэширует ответы: кэш есть только у Experiment в ноутбуке.
        # Пустой контекст сразу даёт REFUSAL без вызова Ollama (время ≈ поиск).
        context = await self.study.retrieve(question, config)
        context = inject_named_into_context(self.study, question, config, context)
        result = await self.study.answer(question, config, context=context, seed=42)
        answer = polish_answer(result.get("answer") or "")
        if is_citation_only(answer) and result.get("used_ollama"):
            retry = await self.study.answer(question, config, context=context, seed=43)
            retry_answer = polish_answer(retry.get("answer") or "")
            if not is_citation_only(retry_answer) and not is_full_refusal(retry_answer):
                result, answer = retry, retry_answer
        if is_citation_only(answer):
            answer = expand_citation_only(answer)
        sources = present_sources(
            result, approach, config.mode if approach == "light" else None, answer,
            corpus=self.resume_texts,
        )
        return {
            "answer": answer,
            "sources": sources,
            "meta": {"label": sources["label"], "model": sources["model"], "approach": sources["approach"]},
            "input_tokens": result.get("input_tokens"),
            "seconds": result.get("seconds"),
        }



def wants_several_candidates(question: str) -> bool:
    """Вопрос просит нескольких людей, сравнение или обобщение, но не одного лучшего."""
    text = (question or "").casefold()
    if re.search(r"больше всего подходит|наиболее подходит", text):
        return False
    if re.search(r"двух разных|сравни\b|какие\b|что кандидаты\b|найди кандидатов", text):
        return True
    return len(named_candidate_numbers(question)) > 1


class MockEngine:
    """Ответы без Ollama и без индекса — только для проверки UI и API."""

    def __init__(self, root: Path | None = None):
        self.root = root or repo_root()
        self.ready = True
        self.light_ready = True
        self.status = "mock"
        self.detail = "Режим HW11_MOCK_LLM: ответы заглушки."
        self.study = None
        self.light = None
        self.ollama = None
        self._resumes = mock_resumes()
        self.resume_texts = mock_corpus_texts()

    def config_for(self, approach: str, mode: str = DEFAULT_MODE):
        if approach not in APPROACHES:
            raise ValueError(f"Неизвестный подход: {approach}")
        if approach == "light" and mode not in MODES:
            raise ValueError(f"Неизвестный режим LightRAG: {mode}")
        return type("Cfg", (), {"method": approach, "mode": mode})()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def answer(self, question: str, approach: str, mode: str = DEFAULT_MODE) -> dict:
        self.config_for(approach, mode)
        lowered = (question or "").casefold()
        refusal = "kubernetes" in lowered or "e-awb" in lowered
        citation_only = "повар" in lowered
        named = [] if refusal or citation_only else resumes_named_in_question(question, self._resumes)
        if refusal:
            raw = "Ответ: В резюме нет достаточных сведений для ответа. [ID]"
            retrieved = [{"id": "R367", "text": "Обязанности повара без срока работы."}]
            trace = {}
        elif citation_only:
            raw = "[R367]"
            retrieved = [{
                "id": "R367",
                "text": "Приготовление блюд, кулинарная обработка продуктов и работа по технологическим картам.",
            }, {"id": "R99", "text": "Этот фрагмент не должен попасть в источники."}]
            trace = {}
            if approach == "long":
                trace["groups"] = {
                    "G0008": {"source_id": "R367", "distance": 5.1, "fragment": "Приготовление блюд."},
                }
            if approach == "light":
                trace["edges"] = [{
                    "from": "Повар", "to": "Технологические карты",
                    "description": "Готовил блюда по картам",
                    "sources": ["R367"],
                }]
        elif wants_several_candidates(question) or len(named) > 1:
            if named and len(named) > 1:
                rows = named
                raw = (
                    f"Кандидат №1 чаще работает напрямую с клиентами: укладка и стрижка волос [{rows[0].id}]. "
                    f"Кандидат №2 больше занят персоналом и организацией работы [{rows[1].id}]."
                )
            elif "столовой" in lowered or "двух разных" in lowered:
                rows = [row for row in self._resumes if row.id in {"R367", "R15"}]
                raw = (
                    "[R367] готовил блюда по технологическим картам. "
                    "[R15] работал с документами и заявками."
                )
            else:
                rows = [row for row in self._resumes if row.id in {"R1", "R15"}]
                raw = (
                    "В [R1] указаны практические навыки работы с клиентами. "
                    "В [R15] есть опыт работы с документами и перевозками."
                )
            retrieved = [{"id": row.id, "text": row.document} for row in rows]
            trace = {}
            if approach == "long" and rows:
                trace["groups"] = {
                    "G0002": {"source_id": rows[0].id, "distance": 0.0, "fragment": (rows[0].document or "")[:80]},
                }
                if len(rows) > 1:
                    trace["groups"]["G0003"] = {
                        "source_id": rows[1].id, "distance": 0.1, "fragment": (rows[1].document or "")[:80],
                    }
            if approach == "light":
                trace["edges"] = []
        elif named:
            row = named[0]
            if row.id == "R1":
                raw = "Кандидат №1 занимается расчесыванием, стрижкой и укладкой волос. [R1]"
            elif row.id == "R18":
                raw = "Кандидат №18 организовывал строительные подряды. [R18]"
            else:
                raw = f"Сведения о кандидате есть в резюме. [{row.id}]"
            retrieved = [{"id": row.id, "text": row.document}]
            trace = {}
            if approach == "long":
                trace["groups"] = {
                    "G0001": {"source_id": row.id, "distance": 0.0, "fragment": (row.document or "")[:80]},
                }
            if approach == "light":
                trace["edges"] = []
        else:
            raw = (
                "Ответ: Кандидат умеет делать химическую завивку и окраску волос. [R1] [ID]"
            )
            retrieved = [
                {"id": "R1", "text": "ие волос химическим способом. Окраска волос."},
                {"id": "R99", "text": "Этот фрагмент не должен попасть в источники, его ответ не цитирует."},
            ]
            trace = {}
            if approach == "long":
                trace["groups"] = {
                    "G0005": {
                        "source_id": "R1",
                        "distance": 6.09,
                        "fragment": "Завивка волос химическим способом. Окраска волос.",
                    }
                }
            if approach == "light":
                trace["edges"] = [{
                    "from": "Окраска волос",
                    "to": "Химические препараты",
                    "description": "Химические препараты используются в процессе окраски волос",
                    "sources": ["R1"],
                }]
        result = {"answer": raw, "trace": trace, "sources": retrieved}
        answer = polish_answer(raw)
        if is_citation_only(answer):
            answer = expand_citation_only(answer)
        sources = present_sources(result, approach, mode if approach == "light" else None, answer,
                                  corpus=self.resume_texts)
        return {
            "answer": answer,
            "sources": sources,
            "meta": {"label": sources["label"], "model": sources["model"], "approach": sources["approach"]},
            "input_tokens": 0,
            "seconds": 0.0,
        }


def create_engine():
    if os.getenv("HW11_MOCK_LLM", "").strip() in {"1", "true", "True", "yes"}:
        return MockEngine()
    return Hw11Engine()
