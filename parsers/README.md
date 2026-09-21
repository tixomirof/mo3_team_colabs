# Парсер резюме Hexlet CV

Сайт: https://cv.hexlet.io/ru

Собирает публичные резюме программистов, убирает ФИО и контакты и ставит один из шести стеков: ASP.NET, React, Python, Java, PHP, Node.js. Смешанный стек и резюме не на русском языке не сохраняются.

Вход на сайт не нужен.

## Запуск

```bash
pip install -r requirements.txt
python hexlet_resume_parser.py --limit 5000 --out resumes.csv
```

`--limit` — сколько подходящих резюме сохранить. `--delay` — пауза между запросами в секундах.
