# Парсер hh.ru

Собирает карточки программистов и ставит один из шести стеков: ASP.NET, React, Python, Java, PHP, Node.js.

- `vacancies` — публичная RSS-выдача и открытые страницы вакансий. Токен не нужен.
- `resumes` — обезличенные резюме через API. Нужен одобренный OAuth-токен работодателя.

Поиск резюме и закрытый API вакансий без заявки недоступны. HTML-вход и сессию сайта парсер не обходит.

## Запуск без токена

```bash
pip install -r requirements.txt
python hh_resume_parser.py --source vacancies --check
python hh_resume_parser.py --source vacancies --per-class 20 --out vacancies.csv
```

На пару хватит `--per-class 20`. RSS отдаёт около 20 карточек на запрос, поэтому больше может не набраться.

## Запуск по резюме

```bash
python hh_resume_parser.py --source resumes --check
python hh_resume_parser.py --source resumes --per-class 100 --out resumes.csv
```

## Переменные в `.env`

| Переменная | Что указать |
|---|---|
| `HH_USER_AGENT` | Имя приложения и почта, например `MO3Parser/1.0 (you@mail.ru)`. Желательно для обоих режимов. |
| `HH_ACCESS_TOKEN` | OAuth-токен с [dev.hh.ru](https://dev.hh.ru). Нужен только для `--source resumes`. |
