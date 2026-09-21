"""Парсер обезличенных резюме программистов с Hexlet CV.

Читает публичный каталог и страницы резюме, убирает ФИО и контакты
и оставляет один из шести стеков:

ASP.NET, React, Python, Java, PHP, Node.js.

Вход, пароль и обход закрытых страниц не используются.
Смешанный стек и резюме не на русском языке в набор не попадают.

Пример:
    python hexlet_resume_parser.py --limit 5000 --out resumes.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL = "https://cv.hexlet.io"
LIST_URL = f"{BASE_URL}/ru"
USER_AGENT = "MO3ResumeParser/1.0 (academic)"
DEFAULT_DELAY_SEC = 0.3
DEFAULT_WORKERS = 3
DEFAULT_LIMIT = 5000
MAX_RETRIES = 4
MIN_TEXT_CHARS = 180
MIN_CYRILLIC_CHARS = 80
MIN_CYRILLIC_SHARE = 0.35

STACKS = ("ASP.NET", "React", "Python", "Java", "PHP", "Node.js")
SECTION_LABELS = {
    "Контакты",
    "Описание",
    "Навыки",
    "Описание проектов",
    "Опыт",
    "Образование",
    "О себе",
}

STACK_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "ASP.NET": (
        re.compile(r"asp\.?\s*net", re.I),
        re.compile(r"\bc\s*#\b", re.I),
        re.compile(r"c\s*sharp", re.I),
        re.compile(r"\bdotnet\b", re.I),
        re.compile(r"\.net\b", re.I),
        re.compile(r"entity\s*framework", re.I),
        re.compile(r"\bblazor\b", re.I),
    ),
    "React": (
        re.compile(r"\breact(?:\.?js)?\b", re.I),
        re.compile(r"\bredux\b", re.I),
        re.compile(r"next\.?js", re.I),
    ),
    "Python": (
        re.compile(r"\bpython\b", re.I),
        re.compile(r"\bdjango\b", re.I),
        re.compile(r"\bflask\b", re.I),
        re.compile(r"\bfastapi\b", re.I),
        re.compile(r"\bpandas\b", re.I),
    ),
    "Java": (
        re.compile(r"\bjava\b", re.I),
        re.compile(r"\bspring\s*boot\b", re.I),
        re.compile(r"\bspring\b", re.I),
        re.compile(r"\bhibernate\b", re.I),
    ),
    "PHP": (
        re.compile(r"\bphp\b", re.I),
        re.compile(r"\blaravel\b", re.I),
        re.compile(r"\bsymfony\b", re.I),
        re.compile(r"\byii2?\b", re.I),
        re.compile(r"\bbitrix\b", re.I),
    ),
    "Node.js": (
        re.compile(r"node\.?\s*js", re.I),
        re.compile(r"\bnodejs\b", re.I),
        re.compile(r"\bexpress(?:\.?js)?\b", re.I),
        re.compile(r"\bnest\.?js\b", re.I),
        re.compile(r"\bnestjs\b", re.I),
    ),
}

JAVA_SCRIPT_RE = re.compile(r"javascript", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{8,}\d)")
URL_RE = re.compile(
    r"(?:(?:https?://|www\.)\S+"
    r"|(?:github\.com|gitlab\.com|bitbucket\.org|linkedin\.com|t\.me|"
    r"vk\.com|hh\.ru/resume|career\.habr\.com)/\S+)",
    re.I,
)
HANDLE_RE = re.compile(r"(?<!\w)@[\w.]{3,}")
NAME_INTRO_RE = re.compile(
    r"(?:меня зовут|мое имя|моё имя|my name is)\s+[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,2}",
    re.I,
)
CITY_RE = re.compile(
    r"прожива(?:ю|ет)\s+в\s+(?:городе\s+)?([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,40})",
    re.I,
)
RESUME_LINK_RE = re.compile(r'href="/ru/resumes/(\d+)"')
PAGE_LINK_RE = re.compile(r'href="/ru\?page=(\d+)"')

CSV_FIELDS = (
    "resume_id",
    "title",
    "skills",
    "experience_text",
    "text",
    "area",
    "stack",
    "stack_score",
)


class CatalogClient:
    """Публичные HTML-страницы с общей паузой между запросами."""

    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = delay_sec
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                }
            )
            self._local.session = session
        return session

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            self._next_at = max(self._next_at, now) + self.delay_sec
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str) -> str | None:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_turn()
            try:
                response = self._session().get(url, timeout=30)
            except requests.RequestException as error:
                last_error = error
                time.sleep(self.delay_sec * attempt)
                continue
            if response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(self.delay_sec * (attempt + 1))
                last_error = RuntimeError(f"Код {response.status_code} для {url}")
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Запрос {url} завершился кодом {response.status_code}")
            response.encoding = "utf-8"
            return response.text
        raise RuntimeError(f"Не удалось открыть {url}: {last_error}")


def list_page_url(page: int) -> str:
    if page <= 1:
        return LIST_URL
    return f"{LIST_URL}?page={page}"


def resume_url(resume_id: str) -> str:
    return f"{BASE_URL}/ru/resumes/{resume_id}"


def discover_resume_ids(client: CatalogClient) -> list[str]:
    """Собираем номера резюме со всех страниц открытого каталога."""
    first_html = client.get(list_page_url(1))
    if not first_html:
        raise RuntimeError("Каталог резюме не открылся")
    page_numbers = [int(value) for value in PAGE_LINK_RE.findall(first_html)]
    last_page = max(page_numbers) if page_numbers else 1
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def absorb(page_html: str) -> None:
        for resume_id in RESUME_LINK_RE.findall(page_html):
            if resume_id not in seen:
                seen.add(resume_id)
                ordered_ids.append(resume_id)

    absorb(first_html)
    print(f"Страниц каталога: {last_page}")
    for page in range(2, last_page + 1):
        page_html = client.get(list_page_url(page))
        if not page_html:
            continue
        absorb(page_html)
        if page % 20 == 0 or page == last_page:
            print(f"  просмотрено страниц: {page}, резюме в списке: {len(ordered_ids)}")
    return ordered_ids


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def anonymize(value: str, person_name: str) -> str:
    """Убираем ФИО, почту, телефон, ссылки и ники."""
    text = clean_text(value)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = NAME_INTRO_RE.sub(" ", text)
    if person_name:
        text = re.sub(re.escape(person_name), " ", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_russian(text: str) -> bool:
    cyrillic = len(re.findall(r"[а-яё]", text, flags=re.I))
    latin = len(re.findall(r"[a-z]", text, flags=re.I))
    letters = cyrillic + latin
    if cyrillic < MIN_CYRILLIC_CHARS or letters == 0:
        return False
    return cyrillic / letters >= MIN_CYRILLIC_SHARE


def prepare_for_java(text: str) -> str:
    return JAVA_SCRIPT_RE.sub(" ", text)


def stack_scores(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for stack, patterns in STACK_PATTERNS.items():
        haystack = prepare_for_java(text) if stack == "Java" else text
        scores[stack] = sum(1 for pattern in patterns if pattern.search(haystack))
    return scores


def classify(title: str, skills: str, body: str) -> tuple[str, int]:
    """Один класс, если стек заметно сильнее остальных."""
    scores = stack_scores(body)
    for stack, score in stack_scores(title).items():
        scores[stack] += score * 2
    for stack, score in stack_scores(skills).items():
        scores[stack] += score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_stack, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score < 2:
        return "", 0
    if second_score >= 2 and best_score - second_score < 2:
        return "", best_score
    return best_stack, best_score


def section_texts(main: Tag) -> dict[str, str]:
    """Делим карточку на блоки по заголовкам, не заходя в чужие комментарии."""
    buckets: dict[str, list[str]] = {label: [] for label in SECTION_LABELS}
    current: str | None = None
    for node in main.descendants:
        if isinstance(node, Tag) and node.name == "h3":
            if "Комментар" in node.get_text(" ", strip=True):
                break
        if isinstance(node, Tag) and node.name in {"h2", "b"}:
            label = node.get_text(" ", strip=True)
            if label in SECTION_LABELS:
                current = label
                continue
        if current is None or not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if not isinstance(parent, Tag) or parent.name in {"h1", "h2", "h3", "b", "script", "style"}:
            continue
        piece = str(node).strip()
        if piece:
            buckets[current].append(piece)
    return {label: clean_text("\n".join(parts)) for label, parts in buckets.items()}


def parse_resume(page_html: str, resume_id: str) -> dict[str, str] | None:
    soup = BeautifulSoup(page_html, "html.parser")
    main = soup.find("main")
    if not isinstance(main, Tag):
        return None
    for side in main.select(".col-lg-3"):
        side.decompose()

    name_node = main.select_one("h1 a")
    title_node = main.select_one("h3.text-center")
    person_name = name_node.get_text(" ", strip=True) if name_node else ""
    title = title_node.get_text(" ", strip=True) if title_node else ""
    sections = section_texts(main)

    skills = anonymize(sections.get("Навыки", ""), person_name)
    experience_text = anonymize(sections.get("Опыт", ""), person_name)
    description_parts = [
        sections.get("Описание", ""),
        sections.get("Описание проектов", ""),
        sections.get("Образование", ""),
        sections.get("О себе", ""),
    ]
    description = anonymize("\n".join(part for part in description_parts if part), person_name)
    title = anonymize(title, person_name)
    city_match = CITY_RE.search(description)
    area = city_match.group(1).strip() if city_match else ""

    body = "\n".join(piece for piece in (title, skills, description, experience_text) if piece)
    if len(body) < MIN_TEXT_CHARS or not is_russian(body):
        return None
    stack, score = classify(title, skills, body)
    if not stack:
        return None
    return {
        "resume_id": resume_id,
        "title": title,
        "skills": skills,
        "experience_text": experience_text,
        "text": body,
        "area": area,
        "stack": stack,
        "stack_score": str(score),
    }


def fetch_resume(client: CatalogClient, resume_id: str) -> dict[str, str] | None:
    try:
        page_html = client.get(resume_url(resume_id))
    except Exception as error:
        print(f"Пропуск резюме {resume_id}: {error}", file=sys.stderr)
        return None
    if not page_html:
        return None
    return parse_resume(page_html, resume_id)


def collect_resumes(client: CatalogClient, limit: int, workers: int) -> list[dict[str, str]]:
    resume_ids = discover_resume_ids(client)
    print(f"Открытых карточек: {len(resume_ids)}. Беру до {limit} подходящих.")
    rows: list[dict[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_resume, client, resume_id) for resume_id in resume_ids]
        for future in as_completed(futures):
            done += 1
            row = future.result()
            if row is not None:
                rows.append(row)
            if done % 100 == 0 or done == len(futures):
                print(f"Обработано {done}/{len(futures)}, в наборе {len(rows)}")
            if len(rows) >= limit:
                break
    rows.sort(key=lambda row: int(row["resume_id"]), reverse=True)
    return rows[:limit]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Парсер обезличенных резюме с Hexlet CV")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Сколько резюме сохранить")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC, help="Пауза между запросами, секунды")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Сколько страниц качать параллельно")
    parser.add_argument("--out", default="", help="Куда сохранить CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0:
        print("Параметр --limit должен быть больше нуля", file=sys.stderr)
        return 1
    if args.delay < 0 or args.workers <= 0:
        print("Пауза и число потоков заданы неверно", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "resumes.csv"
    client = CatalogClient(args.delay)
    try:
        rows = collect_resumes(client, args.limit, args.workers)
    except Exception as error:
        print(f"Ошибка парсера: {error}", file=sys.stderr)
        return 1
    write_csv(out_path, rows)
    counts: dict[str, int] = {stack: 0 for stack in STACKS}
    for row in rows:
        counts[row["stack"]] += 1
    print(f"Готово: {len(rows)} обезличенных резюме -> {out_path}")
    for stack in STACKS:
        print(f"  {stack}: {counts[stack]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
