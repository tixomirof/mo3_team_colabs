"""Парсер обезличенных резюме программистов с hh.ru.

Собирает короткие карточки из официального API поиска резюме,
убирает личные данные и ставит один из шести стеков:

ASP.NET, React, Python, Java, PHP, Node.js.

Логин и пароль не используются. Нужен OAuth-токен работодателя
с доступом к базе резюме. Его получают на https://dev.hh.ru
и кладут в HH_ACCESS_TOKEN.

Пример:
    python hh_resume_parser.py --per-class 100 --out resumes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.hh.ru"
PROGRAMMER_ROLE_ID = "96"
DEFAULT_PER_PAGE = 50
DEFAULT_DELAY_SEC = 0.8
MAX_RETRIES = 4

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
    """Нет прав на поиск резюме или токен не принят."""


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
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": user_agent,
        "HH-User-Agent": user_agent,
        "Accept": "application/json",
    }


class HeadHunterClient:
    """Клиент официального API поиска резюме с паузой и повторными попытками."""

    def __init__(self, token: str, user_agent: str, delay_sec: float) -> None:
        self.session = requests.Session()
        self.session.headers.update(make_headers(token, user_agent))
        self.delay_sec = delay_sec

    def get(self, path: str, params: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(self.delay_sec)
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
            "Нет доступа к поиску резюме. Нужен OAuth-токен работодателя "
            "с услугой доступа к базе резюме. Логин, пароль и обход входа "
            "парсер не использует. "
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

    result: list[dict[str, str]] = []
    for stack, rows in classified.items():
        taken = rows[:per_class]
        result.extend(taken)
        print(f"Класс {stack}: сохранено {len(taken)} резюме")
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
    parser = argparse.ArgumentParser(
        description="Парсер обезличенных резюме программистов с hh.ru"
    )
    parser.add_argument("--token", help="OAuth-токен. Если не указан, берётся HH_ACCESS_TOKEN")
    parser.add_argument("--user-agent", help="Имя приложения и контакт, как требует hh.ru")
    parser.add_argument("--per-class", type=int, default=100, help="Сколько резюме на класс")
    parser.add_argument("--area", help="Код региона, например 1 для Москвы")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC, help="Пауза между запросами")
    parser.add_argument("--out", default="resumes.csv", help="Путь к CSV")
    parser.add_argument("--json-out", help="Дополнительно сохранить JSON")
    parser.add_argument("--check", action="store_true", help="Только проверить токен через /me")
    return parser.parse_args(argv)


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    token = args.token or os.getenv("HH_ACCESS_TOKEN", "").strip()
    user_agent = (
        args.user_agent
        or os.getenv("HH_USER_AGENT", "").strip()
        or "MO3ResumeParser/1.0 (student@mail.ru)"
    )
    if not token:
        raise AccessError(
            "Не задан токен. Создайте приложение на https://dev.hh.ru, "
            "дождитесь одобрения и запишите OAuth-токен работодателя "
            "в HH_ACCESS_TOKEN. Пароль аккаунта не нужен."
        )
    return token, user_agent


def main(argv: list[str] | None = None) -> int:
    load_settings()
    args = parse_args(argv)
    try:
        token, user_agent = resolve_credentials(args)
        client = HeadHunterClient(token, user_agent, args.delay)
        if args.check:
            profile = client.check_token()
            auth_type = profile.get("auth_type") or profile.get("is_employer")
            print(f"Токен принят. Тип доступа: {auth_type}")
            return 0
        if args.per_class <= 0:
            raise ValueError("Параметр --per-class должен быть больше нуля")
        rows = collect_resumes(client, args.per_class, args.area, DEFAULT_PER_PAGE)
        out_path = Path(args.out)
        write_csv(out_path, rows)
        if args.json_out:
            write_json(Path(args.json_out), rows)
        print(f"Готово: {len(rows)} обезличенных резюме -> {out_path}")
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
