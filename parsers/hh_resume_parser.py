"""Парсер карточек программистов с hh.ru.

Два режима:

* vacancies — публичная RSS-выдача и открытые страницы вакансий, токен не нужен;
* resumes — обезличенные резюме через API, нужен OAuth-токен работодателя.

Классы стека: ASP.NET, React, Python, Java, PHP, Node.js.

Примеры:
    python hh_resume_parser.py --source vacancies --per-class 20 --out vacancies.csv
    python hh_resume_parser.py --source resumes --per-class 100 --out resumes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

API_BASE = "https://api.hh.ru"
RSS_URL = "https://hh.ru/search/vacancy/rss"
VACANCY_URL = "https://hh.ru/vacancy/{vacancy_id}"
PROGRAMMER_ROLE_ID = "96"
DEFAULT_PER_PAGE = 50
DEFAULT_DELAY_SEC = 0.8
MAX_RETRIES = 4

# Не больше шести классов. Для RSS несколько запросов, потому что лента короткая.
STACK_SEARCHES = {
    "ASP.NET": ("ASP.NET", "C#", "ASP.NET Core"),
    "React": ("React", "React.js", "Next.js"),
    "Python": ("Python", "Django", "FastAPI"),
    "Java": ("Java Spring", "Java developer"),
    "PHP": ("PHP", "Laravel"),
    "Node.js": ("Node.js", "NestJS"),
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
        re.compile(r"\bwpf\b", re.I),
        re.compile(r"\bwinforms\b", re.I),
        re.compile(r"ado\.net", re.I),
    ),
    "React": (
        re.compile(r"\breact(?:\.?js)?\b", re.I),
        re.compile(r"\bredux\b", re.I),
        re.compile(r"next\.?js", re.I),
        re.compile(r"\breact-dom\b", re.I),
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
        re.compile(r"\bjakarta\b", re.I),
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
HIGHLIGHT_RE = re.compile(r"</?highlighttext>", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
JSON_LD_RE = re.compile(
    r'<script type="application/ld\+json">(.*?)</script>',
    re.I | re.S,
)
VACANCY_ID_RE = re.compile(r"/vacancy/(\d+)")
SKILL_QA_RE = re.compile(
    r'data-qa="skills-element"[^>]*>([^<]+)',
    re.I,
)
CSV_FIELDS = (
    "resume_id",
    "title",
    "skills",
    "experience_text",
    "text",
    "area",
    "experience_months",
    "stack",
    "stack_score",
)


class AccessError(RuntimeError):
    """Нет прав на выбранный источник."""


def load_env_file(path: Path) -> None:
    """Читаем KEY=VALUE из .env, не перезаписывая уже заданные переменные."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> None:
    parser_dir = Path(__file__).resolve().parent
    load_env_file(parser_dir / ".env")
    load_env_file(Path.cwd() / ".env")


def make_headers(token: str, user_agent: str) -> dict[str, str]:
    headers = {
        "User-Agent": user_agent,
        "HH-User-Agent": user_agent,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class HeadHunterClient:
    """Клиент hh.ru: API резюме, публичная RSS и открытые страницы вакансий."""

    def __init__(self, token: str, user_agent: str, delay_sec: float) -> None:
        self.session = requests.Session()
        self.session.headers.update(make_headers(token, user_agent))
        self.delay_sec = delay_sec

    def _sleep(self) -> None:
        time.sleep(self.delay_sec)

    def get(self, path: str, params: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._sleep()
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as error:
                last_error = error
                time.sleep(self.delay_sec * attempt)
                continue

            if response.status_code == 429:
                time.sleep(self.delay_sec * (attempt + 1))
                continue
            if response.status_code in {401, 403}:
                raise AccessError(self._access_message(response))
            if response.status_code >= 500:
                last_error = RuntimeError(f"Сервер hh.ru вернул {response.status_code}")
                time.sleep(self.delay_sec * attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Запрос {path} завершился кодом {response.status_code}: {response.text[:500]}"
                )
            return response.json()

        raise RuntimeError(f"Не удалось выполнить запрос {path}: {last_error}")

    @staticmethod
    def _access_message(response: requests.Response) -> str:
        details = response.text[:400]
        return (
            "Нет доступа к API резюме. Нужен одобренный OAuth-токен работодателя. "
            "Пока заявку не приняли, запускайте --source vacancies. "
            f"Ответ сервера ({response.status_code}): {details}"
        )

    def check_token(self) -> dict[str, Any]:
        return self.get("/me")

    def search_resumes(self, query: str, page: int, per_page: int, area: str | None) -> dict[str, Any]:
        params: list[tuple[str, str]] = [
            ("text", query),
            ("professional_role", PROGRAMMER_ROLE_ID),
            ("per_page", str(per_page)),
            ("page", str(page)),
            ("order_by", "relevance"),
        ]
        if area:
            params.append(("area", area))
        return self.get("/resumes", params=params)

    def search_vacancy_rss(self, query: str, area: str | None) -> list[dict[str, str]]:
        params = {
            "text": query,
            "professional_role": PROGRAMMER_ROLE_ID,
        }
        if area:
            params["area"] = area
        url = f"{RSS_URL}?{urlencode(params)}"
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._sleep()
            try:
                response = self.session.get(
                    url,
                    timeout=30,
                    headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
                )
            except requests.RequestException as error:
                last_error = error
                time.sleep(self.delay_sec * attempt)
                continue
            if response.status_code == 429:
                time.sleep(self.delay_sec * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"RSS поиска вакансий вернул {response.status_code}: {response.text[:300]}"
                )
            return parse_rss_items(response.content)
        raise RuntimeError(f"Не удалось прочитать RSS: {last_error}")

    def fetch_vacancy_html(self, vacancy_id: str) -> str:
        url = VACANCY_URL.format(vacancy_id=vacancy_id)
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._sleep()
            try:
                response = self.session.get(
                    url,
                    timeout=30,
                    headers={"Accept": "text/html,application/xhtml+xml"},
                )
            except requests.RequestException as error:
                last_error = error
                time.sleep(self.delay_sec * attempt)
                continue
            if response.status_code == 429:
                time.sleep(self.delay_sec * (attempt + 1))
                continue
            if response.status_code in {404, 410}:
                return ""
            if response.status_code >= 400:
                last_error = RuntimeError(f"Страница вакансии {vacancy_id}: {response.status_code}")
                continue
            return response.text
        raise RuntimeError(f"Не удалось открыть вакансию {vacancy_id}: {last_error}")


def parse_rss_items(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    items: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        link = (item.findtext("link") or "").strip()
        match = VACANCY_ID_RE.search(link)
        if not match:
            continue
        items.append(
            {
                "id": match.group(1),
                "title": (item.findtext("title") or "").strip(),
                "description": html_to_text(item.findtext("description") or ""),
                "link": link,
            }
        )
    return items


def html_to_text(value: str) -> str:
    text = HIGHLIGHT_RE.sub("", value)
    text = HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_jobposting(html: str) -> dict[str, Any]:
    for match in JSON_LD_RE.finditer(html):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") == "JobPosting":
                return candidate
    return {}


def parse_vacancy_page(item: dict[str, str], html: str) -> dict[str, str]:
    posting = extract_jobposting(html) if html else {}
    title = str(posting.get("title") or item.get("title") or "").strip()
    description = html_to_text(str(posting.get("description") or item.get("description") or ""))
    skills = "; ".join(SKILL_QA_RE.findall(html)) if html else ""
    area = ""
    job_location = posting.get("jobLocation")
    if isinstance(job_location, dict):
        address = job_location.get("address")
        if isinstance(address, dict):
            area = str(address.get("addressLocality") or address.get("addressRegion") or "").strip()
    text = "\n".join(piece for piece in (title, skills, description) if piece)
    return {
        "resume_id": str(item.get("id") or "").strip(),
        "title": title,
        "skills": skills,
        "experience_text": description,
        "text": text,
        "area": area,
        "experience_months": "",
    }


def join_skills(item: dict[str, Any]) -> str:
    skills = item.get("skill_set") or []
    if isinstance(skills, str):
        return skills.strip()
    names = []
    for skill in skills:
        if isinstance(skill, str) and skill.strip():
            names.append(skill.strip())
        elif isinstance(skill, dict):
            name = str(skill.get("name") or "").strip()
            if name:
                names.append(name)
    extra = item.get("skills")
    if isinstance(extra, str) and extra.strip():
        names.append(extra.strip())
    unique = list(dict.fromkeys(names))
    return "; ".join(unique)


def join_experience(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for job in item.get("experience") or []:
        if not isinstance(job, dict):
            continue
        position = str(job.get("position") or "").strip()
        description = str(job.get("description") or "").strip()
        chunk = ". ".join(piece for piece in (position, description) if piece)
        if chunk:
            parts.append(chunk)
    return "\n".join(parts)


def experience_months(item: dict[str, Any]) -> str:
    total = item.get("total_experience") or {}
    if isinstance(total, dict) and total.get("months") is not None:
        return str(total["months"])
    return ""


def anonymize(item: dict[str, Any]) -> dict[str, str]:
    """Оставляем только поля без ФИО, контактов и фото."""
    title = str(item.get("title") or "").strip()
    skills = join_skills(item)
    experience_text = join_experience(item)
    area = ""
    if isinstance(item.get("area"), dict):
        area = str(item["area"].get("name") or "").strip()
    text = "\n".join(piece for piece in (title, skills, experience_text) if piece)
    return {
        "resume_id": str(item.get("id") or "").strip(),
        "title": title,
        "skills": skills,
        "experience_text": experience_text,
        "text": text,
        "area": area,
        "experience_months": experience_months(item),
    }


def prepare_for_java(text: str) -> str:
    return JAVA_SCRIPT_RE.sub(" ", text)


def stack_scores(text: str) -> dict[str, int]:
    scores = {}
    for stack, patterns in STACK_PATTERNS.items():
        haystack = prepare_for_java(text) if stack == "Java" else text
        scores[stack] = sum(1 for pattern in patterns if pattern.search(haystack))
    return scores


def classify(record: dict[str, str], search_stack: str | None = None) -> tuple[str, int]:
    """Выбираем стек по тексту; поисковый запрос даёт лишь небольшой бонус."""
    scores = stack_scores(record["text"])
    title_scores = stack_scores(record["title"])
    skill_scores = stack_scores(record["skills"])
    for stack in scores:
        scores[stack] += title_scores[stack] + skill_scores[stack]
        if search_stack == stack:
            scores[stack] += 1

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_stack, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score < 2:
        return "", 0
    if second_score >= 2 and best_score - second_score < 2:
        return "", best_score
    return best_stack, best_score


def collect_resumes(
    client: HeadHunterClient,
    per_class: int,
    area: str | None,
    per_page: int,
) -> list[dict[str, str]]:
    raw_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for stack, queries in STACK_SEARCHES.items():
        gathered = 0
        print(f"Ищу резюме для класса {stack}...")
        for query in queries:
            page = 0
            while gathered < per_class:
                payload = client.search_resumes(query, page, per_page, area)
                items = payload.get("items") or []
                if not items:
                    break
                for item in items:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    resume_id = str(item["id"])
                    if resume_id not in raw_by_id:
                        raw_by_id[resume_id] = (stack, item)
                        gathered += 1
                        if gathered >= per_class:
                            break
                pages = int(payload.get("pages") or 0)
                page += 1
                if page >= pages:
                    break
            if gathered >= per_class:
                break
        print(f"  карточек после поиска: {gathered}")

    classified: dict[str, list[dict[str, str]]] = {stack: [] for stack in STACK_SEARCHES}
    for search_stack, item in raw_by_id.values():
        record = anonymize(item)
        if not record["resume_id"] or not record["text"]:
            continue
        stack, score = classify(record, search_stack)
        if not stack:
            continue
        record["stack"] = stack
        record["stack_score"] = str(score)
        classified[stack].append(record)
    return take_per_class(classified, per_class)


def collect_vacancies(
    client: HeadHunterClient,
    per_class: int,
    area: str | None,
) -> list[dict[str, str]]:
    raw_by_id: dict[str, tuple[str, dict[str, str]]] = {}
    for stack, queries in STACK_SEARCHES.items():
        gathered = 0
        print(f"Ищу вакансии для класса {stack}...")
        for query in queries:
            if gathered >= per_class:
                break
            for item in client.search_vacancy_rss(query, area):
                vacancy_id = item["id"]
                if vacancy_id in raw_by_id:
                    continue
                raw_by_id[vacancy_id] = (stack, item)
                gathered += 1
                if gathered >= per_class:
                    break
        print(f"  карточек после RSS: {gathered}")

    classified: dict[str, list[dict[str, str]]] = {stack: [] for stack in STACK_SEARCHES}
    print(f"Открываю {len(raw_by_id)} публичных страниц вакансий...")
    for search_stack, item in raw_by_id.values():
        html = client.fetch_vacancy_html(item["id"])
        record = parse_vacancy_page(item, html)
        if not record["resume_id"] or not record["text"]:
            continue
        stack, score = classify(record, search_stack)
        if not stack:
            # В RSS мало текста, поэтому если страница не противоречит запросу — берём его класс.
            stack, score = search_stack, 1
        record["stack"] = stack
        record["stack_score"] = str(score)
        classified[stack].append(record)
    return take_per_class(classified, per_class)


def take_per_class(
    classified: dict[str, list[dict[str, str]]],
    per_class: int,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for stack, rows in classified.items():
        taken = rows[:per_class]
        result.extend(taken)
        print(f"Класс {stack}: сохранено {len(taken)}")
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Парсер карточек программистов с hh.ru")
    parser.add_argument(
        "--source",
        choices=("vacancies", "resumes"),
        default="vacancies",
        help="vacancies — публичные вакансии без токена; resumes — резюме по OAuth",
    )
    parser.add_argument("--token", help="OAuth-токен. Если не указан, берётся HH_ACCESS_TOKEN")
    parser.add_argument("--user-agent", help="Имя приложения и контакт")
    parser.add_argument("--per-class", type=int, default=20, help="Сколько записей на класс")
    parser.add_argument("--area", help="Код региона, например 1 для Москвы")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC, help="Пауза между запросами")
    parser.add_argument("--out", default="dataset.csv", help="Путь к CSV")
    parser.add_argument("--json-out", help="Дополнительно сохранить JSON")
    parser.add_argument("--check", action="store_true", help="Проверить доступ к выбранному источнику")
    return parser.parse_args(argv)


def resolve_user_agent(args: argparse.Namespace) -> str:
    return (
        args.user_agent
        or os.getenv("HH_USER_AGENT", "").strip()
        or "MO3Parser/1.0 (academic vacancy parser)"
    )


def resolve_token(args: argparse.Namespace, required: bool) -> str:
    token = args.token or os.getenv("HH_ACCESS_TOKEN", "").strip()
    if required and not token:
        raise AccessError(
            "Для резюме нужен OAuth-токен с https://dev.hh.ru. "
            "До одобрения заявки запускайте парсер с --source vacancies."
        )
    return token


def main(argv: list[str] | None = None) -> int:
    load_settings()
    args = parse_args(argv)
    try:
        user_agent = resolve_user_agent(args)
        token = resolve_token(args, required=(args.source == "resumes"))
        client = HeadHunterClient(token, user_agent, args.delay)
        if args.check:
            if args.source == "resumes":
                profile = client.check_token()
                auth_type = profile.get("auth_type") or profile.get("is_employer")
                print(f"Токен принят. Тип доступа: {auth_type}")
            else:
                items = client.search_vacancy_rss("Python", args.area)
                print(f"Публичная RSS-выдача доступна. Карточек в ленте: {len(items)}")
            return 0
        if args.per_class <= 0:
            raise ValueError("Параметр --per-class должен быть больше нуля")
        if args.source == "resumes":
            rows = collect_resumes(client, args.per_class, args.area, DEFAULT_PER_PAGE)
            kind = "обезличенных резюме"
        else:
            rows = collect_vacancies(client, args.per_class, args.area)
            kind = "вакансий"
        out_path = Path(args.out)
        write_csv(out_path, rows)
        if args.json_out:
            write_json(Path(args.json_out), rows)
        print(f"Готово: {len(rows)} {kind} -> {out_path}")
        print("Классы:", ", ".join(STACK_SEARCHES))
        return 0
    except AccessError as error:
        print(error, file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Ошибка парсера: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
