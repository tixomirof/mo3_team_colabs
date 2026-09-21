# Парсер резюме hh.ru

Собирает обезличенные карточки резюме программистов через официальный API hh.ru, убирает ФИО и контакты, ставит один из шести стеков: ASP.NET, React, Python, Java, PHP, Node.js. Результат пишется в CSV.

Нужен аккаунт работодателя с доступом к базе резюме. Логин и пароль не используются.

## Запуск

```bash
pip install -r requirements.txt
python hh_resume_parser.py --check
python hh_resume_parser.py --per-class 100 --out resumes.csv
```

`--check` только проверяет токен. `--per-class` — сколько резюме брать на каждый стек.

## Переменные в `.env`

| Переменная | Что указать |
|---|---|
| `HH_ACCESS_TOKEN` | OAuth-токен приложения с [dev.hh.ru](https://dev.hh.ru), не пароль аккаунта |
| `HH_USER_AGENT` | Имя приложения и контактная почта, например `MO3ResumeParser/1.0 (you@example.com)` |
