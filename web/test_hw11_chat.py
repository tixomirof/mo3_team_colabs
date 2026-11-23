"""Проверки селектора чата без Ollama и без тяжёлой индексации."""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent


def test_syntax():
    for folder in (WEB, ROOT / "ml_hw11", ROOT / "labs" / "ml_hw11"):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_polish_and_citations():
    sys.path.insert(0, str(WEB))
    from hw11_backend import (
        CHAT_INSTRUCTION, REFUSAL, cited_ids, expand_citation_only, is_citation_only,
        is_full_refusal, chat_model, polish_answer, present_sources, SNIPPET_LIMIT, _preview,
    )
    assert "каждого подходящего" in CHAT_INSTRUCTION or "двух разные роли" in CHAT_INSTRUCTION or "сравни" in CHAT_INSTRUCTION
    assert "кто больше всего подходит" in CHAT_INSTRUCTION

    assert polish_answer("Ответ: Кандидат умеет делать завивку. [R1] [ID]") == (
        "Кандидат умеет делать завивку [R1]."
    )
    assert polish_answer("Подходит кандидат [R367]").endswith(".")
    assert polish_answer("мытье головы. [R1]") == "мытье головы [R1]."
    assert polish_answer("мытье головы. [R1].") == "мытье головы [R1]."
    assert polish_answer("Наиболее подходящим кандидатом является Р141, опыт есть.") == (
        "Наиболее подходящим кандидатом является [R141], опыт есть."
    )
    assert polish_answer("Наиболее подходящий кандидат — Р[338], опыт есть.") == (
        "Наиболее подходящий кандидат — [R338], опыт есть."
    )
    assert polish_answer("Подходит **R353**, опыт есть.") == "Подходит [R353], опыт есть."
    assert "Р141" not in polish_answer("является Р141, опыт есть")
    assert "Р[338]" not in polish_answer("кандидат — Р[338], опыт есть")
    assert _preview("ие кулинарной обработки продуктов и работа по картам").startswith("кулинарной")
    assert _preview("Приготовление блюд по технологическим картам").startswith("Приготовление")
    assert polish_answer("В резюме нет достаточных сведений для ответа. [ID]") == REFUSAL
    assert polish_answer("Ответ: В резюме нет достаточных сведений для ответа.") == REFUSAL
    mixed = "В резюме нет достаточных сведений для полного стажа, но опыт повара есть. [R367]"
    assert polish_answer(mixed) == (
        "В резюме нет достаточных сведений для полного стажа, но опыт повара есть [R367]."
    )
    assert not is_full_refusal(mixed)
    assert is_citation_only("[R367]")
    assert expand_citation_only("[R367]") == "Наиболее подходящий кандидат — [R367]."
    assert cited_ids("Подходят [R35] и R110, не [ID].") == ["R35", "R110"]
    assert cited_ids("[ID R308] и [R308]") == ["R308"]
    assert cited_ids("является Р141, опыт есть") == ["R141"]
    assert cited_ids("кандидат — Р[338], поскольку") == ["R338"]
    assert cited_ids("опыт есть у **R353**") == ["R353"]
    compare = (
        "Кандидат №1 имеет более широкий опыт, включая мытье головы. "
        "Кандидат №2 работает с персоналом, по сравнению с кандидатом №1. [R1]."
    )
    assert cited_ids(compare) == ["R1", "R2"]
    assert cited_ids(polish_answer(compare)) == ["R1", "R2"]

    raw = {
        "sources": [
            {"id": "R1", "text": "Кандидат №1. Химическая завивка и окраска волос. " + ("слово " * 80)},
            {"id": "R99", "text": "Этот фрагмент не цитируется."},
        ],
        "trace": {
            "groups": {"G0005": {"source_id": "R1", "distance": 6.09, "fragment": "завивка"}},
            "edges": [{
                "from": "Окраска волос", "to": "Химические препараты",
                "description": "длинное описание, его не нужно целиком",
                "sources": ["R1"],
            }],
        },
    }
    answer = "Кандидат умеет делать химическую завивку и окраску волос. [R1]"
    long_payload = present_sources(raw, "long", None, answer)
    assert long_payload["model"] == chat_model()
    assert long_payload["label"] == "LongRAG"
    assert long_payload["used"] is True
    assert [item["id"] for item in long_payload["items"]] == ["R1"]
    assert long_payload["items"][0]["group_id"] == "G0005"
    assert len(long_payload["items"][0]["snippet"]) <= SNIPPET_LIMIT
    assert "R99" not in json.dumps(long_payload, ensure_ascii=False)
    assert "groups" not in long_payload
    assert "resumes" not in long_payload

    light_payload = present_sources(raw, "light", "mix", answer)
    assert light_payload["items"][0]["relations"] == ["Окраска волос → Химические препараты"]

    empty = present_sources(raw, "baseline", None, REFUSAL)
    assert empty["used"] is False
    assert empty["items"] == []
    assert empty["label"] == "Базовая RAG"
    assert empty["model"] == chat_model()

    from hw11_backend import mock_corpus_texts
    corpus = mock_corpus_texts()
    assert corpus["C15"]["id"] == "R15"
    chunked = {
        "sources": [{"id": "R1", "text": "ие волос химическим способом. Окраска волос."}],
        "trace": raw["trace"],
    }
    with_full = present_sources(chunked, "light", "mix", answer, corpus=corpus)
    assert with_full["items"][0]["id"] == "R1"
    assert with_full["items"][0]["full_text"].startswith("Кандидат №1")
    assert "обесцвечивание" in with_full["items"][0]["full_text"].casefold()
    assert len(with_full["items"][0]["full_text"]) > len(with_full["items"][0]["snippet"])
    assert with_full["items"][0]["snippet"].startswith("волос")
    assert with_full["items"][0]["full_text"] != with_full["items"][0]["snippet"]

    missed = present_sources(
        {"sources": [{"id": "R1", "text": "Кандидат №1. Стрижка и мытье головы."}], "trace": {}},
        "long", None, polish_answer(compare), corpus=corpus,
    )
    assert [item["id"] for item in missed["items"]] == ["R1", "R2"]
    assert "персоналом" in missed["items"][1]["full_text"].casefold() or "клиент" in missed["items"][1]["full_text"].casefold()


async def _mock_answers():
    sys.path.insert(0, str(WEB))
    from hw11_backend import REFUSAL, MockEngine, SNIPPET_LIMIT

    engine = MockEngine()
    await engine.start()
    base = await engine.answer("Кто умеет делать завивку?", "baseline")
    long = await engine.answer("Кто умеет делать завивку?", "long")
    light = await engine.answer("Кто умеет делать завивку?", "light", "naive")
    assert "[ID]" not in base["answer"]
    assert base["answer"].startswith("Кандидат умеет")
    assert "Ответ:" not in base["answer"]
    assert base["sources"]["label"] == "Базовая RAG"
    assert "qwen3:4b-instruct" in base["sources"]["model"]
    assert [item["id"] for item in base["sources"]["items"]] == ["R1"]
    assert long["sources"]["items"][0]["group_id"] == "G0005"
    assert len(long["sources"]["items"][0]["snippet"]) <= SNIPPET_LIMIT
    assert light["sources"]["mode"] == "naive"
    assert light["sources"]["items"][0]["relations"]
    assert "Кандидат №1" in base["sources"]["items"][0]["full_text"]
    assert "\n" in base["sources"]["items"][0]["full_text"]
    assert "обесцвечивание" in light["sources"]["items"][0]["full_text"].casefold()
    assert light["sources"]["items"][0]["full_text"] != light["sources"]["items"][0]["snippet"]
    cook = await engine.answer("Кто больше всего подходит на роль повара?", "long")
    assert cook["answer"] != "[R367]"
    assert "Наиболее подходящий кандидат" in cook["answer"]
    assert "[R367]" in cook["answer"]
    assert cook["answer"].endswith(".")
    assert [item["id"] for item in cook["sources"]["items"]] == ["R367"]
    assert cook["sources"]["items"][0]["group_id"] == "G0008"
    assert "Приготовление блюд" in cook["sources"]["items"][0]["full_text"]
    assert "санитарных норм" in cook["sources"]["items"][0]["full_text"]
    none = await engine.answer("Кто администрировал Kubernetes?", "baseline")
    assert none["answer"] == REFUSAL
    assert none["sources"]["used"] is False
    assert none["sources"]["items"] == []
    assert none["sources"]["label"] == "Базовая RAG"
    assert none["sources"]["model"]
    assert none["meta"]["label"] == "Базовая RAG"
    assert none["meta"]["model"] == none["sources"]["model"]
    compare_q = "Сравни кандидатов №1 и №2: у кого больше опыта работы с людьми?"
    for approach in ("baseline", "long", "light"):
        compared = await engine.answer(compare_q, approach, "mix")
        assert "[R1]" in compared["answer"] and "[R2]" in compared["answer"]
        assert [item["id"] for item in compared["sources"]["items"]] == ["R1", "R2"]
        assert "наиболее подходящ" not in compared["answer"].casefold()
    multi = await engine.answer(
        "Кого из двух разных кандидатов подобрать для столовой, если один будет готовить по "
        "технологическим картам, а второй — составлять меню-раскладки и заявки на продукты?",
        "long",
    )
    assert "[R367]" in multi["answer"] and "[R15]" in multi["answer"]
    assert {item["id"] for item in multi["sources"]["items"]} == {"R367", "R15"}
    summary = await engine.answer(
        "Какие форматы коррекционно-развивающих занятий и какие виды психологических тренингов встречаются в резюме?",
        "baseline",
    )
    assert len(summary["sources"]["items"]) >= 2
    best = await engine.answer("Кто больше всего подходит на роль повара?", "long")
    assert best["sources"]["items"][0]["id"] == "R367"
    assert best["answer"].count("[R") == 1 or len(best["sources"]["items"]) == 1
    for approach in ("baseline", "long", "light"):
        about_first = await engine.answer("Кандидат №1 чем занимается", approach, "mix")
        assert about_first["sources"]["items"][0]["id"] == "R1"
        assert "[R1]" in about_first["answer"]
        assert "R19" not in about_first["answer"]
        assert about_first["sources"]["items"][0]["full_text"].startswith("Кандидат №1")
        assert "Расчесывание" in about_first["sources"]["items"][0]["full_text"] or "стрижк" in about_first["sources"]["items"][0]["full_text"].casefold()
        first_word = await engine.answer("Чем занимается первый кандидат", approach, "mix")
        assert first_word["sources"]["items"][0]["id"] == "R1"
        assert "[R1]" in first_word["answer"]
        assert first_word["sources"]["items"][0]["id"] != "R18"
        assert first_word["sources"]["items"][0]["id"] != "R36"
    try:
        await engine.answer("q", "unknown")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    await engine.close()


def test_id_mapping_and_named_candidates():
    sys.path.insert(0, str(WEB))
    from hw11_backend import (
        corpus_texts, mock_corpus_texts, mock_resumes, named_candidate_numbers,
        present_sources, resume_by_candidate_number, resumes_named_in_question,
    )

    assert named_candidate_numbers("Кандидат №1 чем занимается") == [1]
    assert named_candidate_numbers("Чем занимается первый кандидат") == [1]
    assert named_candidate_numbers("что умеет кандидат 18") == [18]
    assert named_candidate_numbers("Кто умеет делать завивку?") == []

    r1 = type("R", (), {
        "id": "R1", "key": "C01", "index_id": "C01", "candidate": 1,
        "category": "Медицина и здравоохранение",
        "document": "Кандидат №1\nРасчесывание, стрижка волос взрослых и детей.",
        "text": "Расчесывание, стрижка волос взрослых и детей.",
    })()
    r18 = type("R", (), {
        "id": "R18", "key": "C18", "index_id": "C18", "candidate": 18,
        "category": "Финансы и бухгалтерия",
        "document": "Кандидат №18\nОрганизовывал строительные подряды.",
        "text": "Организовывал строительные подряды.",
    })()
    r19 = type("R", (), {
        "id": "R19", "key": "R1", "index_id": "R1", "candidate": None,
        "category": "Другое",
        "document": "Резюме R1\nЧужой текст исходного индекса 1.",
        "text": "Чужой текст исходного индекса 1.",
    })()
    r36 = type("R", (), {
        "id": "R36", "key": "R18", "index_id": "R18", "candidate": None,
        "category": "Другое",
        "document": "Резюме R18\nЧужой текст исходного индекса 18.",
        "text": "Чужой текст исходного индекса 18.",
    })()
    rows = [r1, r18, r19, r36]
    table = corpus_texts(rows, root=None)
    assert table["R1"]["id"] == "R1"
    assert table["C01"]["id"] == "R1"
    assert "Расчесывание" in table["R1"]["full_text"]
    assert table["R19"]["id"] == "R19"
    assert "Чужой текст" in table["R19"]["full_text"]
    assert "Резюме R19" in table["R19"]["full_text"]
    assert table["R18"]["id"] == "R18"
    assert table["R36"]["id"] == "R36"
    assert resume_by_candidate_number(rows, 1) is r1
    assert resume_by_candidate_number(rows, 18) is r18
    assert [row.id for row in resumes_named_in_question("Кандидат №1 чем занимается", rows)] == ["R1"]
    assert [row.id for row in resumes_named_in_question("Чем занимается первый кандидат", rows)] == ["R1"]

    payload = present_sources(
        {"sources": [{"id": "R1", "text": "Кандидат №1 Расчесывание, стрижка волос взрослых и детей."}], "trace": {}},
        "baseline", None, "Кандидат №1 занимается стрижкой волос. [R1]", corpus=table,
    )
    assert payload["items"][0]["id"] == "R1"
    assert "Расчесывание" in payload["items"][0]["full_text"]
    assert payload["items"][0]["id"] != "R19"

    light = present_sources(
        {"sources": [{"id": "R18", "text": "Кандидат №18 Организовывал строительные подряды."}], "trace": {}},
        "light", "mix", "Первый кандидат (R18) занимается подрядами. [R18]", corpus=table,
    )
    assert light["items"][0]["id"] == "R18"
    assert "подряды" in light["items"][0]["full_text"].casefold() or "подряды" in light["items"][0]["snippet"].casefold()

    corpus = mock_corpus_texts()
    assert corpus["R1"]["id"] == "R1"
    assert "Расчесывание" in corpus["R1"]["full_text"] or "стрижк" in corpus["R1"]["full_text"].casefold()
    assert corpus["R19"]["id"] == "R19"
    assert corpus["C15"]["id"] == "R15"
    assert [row.id for row in resumes_named_in_question("Кандидат №1 чем занимается", mock_resumes())] == ["R1"]


def test_raw_resume_files():
    import tempfile
    sys.path.insert(0, str(WEB))
    from hw11_backend import corpus_texts, read_raw_resume

    previous = os.environ.pop("HW11_DATA_DIR", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data" / "ML-HW11" / "resumes_selected_raw"
            folder.mkdir(parents=True)
            (folder / "candidate_015_transport.txt").write_text(
                "Полный текст R15 из файла.\nВторая строка.", encoding="utf-8"
            )
            (folder / "candidate_367_trade.txt").write_text("Полный текст повара.", encoding="utf-8")
            assert "Полный текст R15" in (read_raw_resume(root, 15) or "")
            assert read_raw_resume(root, 0) is None
            from_csv = type("R", (), {
                "id": "R15", "key": "C15", "index_id": "C15", "candidate": 15,
                "category": "Транспорт и логистика", "document": "текст из csv",
            })()
            table = corpus_texts([from_csv], root)
            assert table["C15"]["id"] == "R15"
            assert table["C15"]["full_text"] == "текст из csv"
            from_file = type("R", (), {
                "id": "R15", "key": "C15", "index_id": "C15", "candidate": 15,
                "category": "Транспорт и логистика", "document": "", "text": "",
            })()
            table = corpus_texts([from_file], root)
            assert table["C15"]["full_text"].startswith("Полный текст R15")
            assert table["R15"]["full_text"] == table["C15"]["full_text"]
    finally:
        if previous is not None:
            os.environ["HW11_DATA_DIR"] = previous


def test_mock_engine():
    import asyncio
    asyncio.run(_mock_answers())


def _sse_events(body: bytes) -> list[dict]:
    text = body.decode("utf-8")
    events = []
    for block in text.split("\n\n"):
        line = next((row for row in block.split("\n") if row.startswith("data: ")), None)
        if line:
            events.append(json.loads(line[6:]))
    return events


def test_api_mock():
    os.environ["HW11_MOCK_LLM"] = "1"
    os.environ["RAG_ENGINE"] = "auto"
    os.chdir(WEB)
    if str(WEB) not in sys.path:
        sys.path.insert(0, str(WEB))
    import importlib
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        health = client.get("/health").json()
        assert health["model"] == "qwen3:4b-instruct"
        assert health["hw11"]["status"] == "mock"
        assert health["hw11"]["ready"] is True
        assert health["rag_engine"] == "hw11"

        page = client.get("/")
        assert page.status_code == 200
        assert 'id="approach"' in page.text
        assert "Базовая RAG" in page.text
        assert "LongRAG" in page.text
        assert "LightRAG" in page.text
        assert 'value="mix" selected' in page.text
        assert "Источники (' + items.length + ')'" in page.text or "Источники (" in page.text
        assert "createElement('details')" in page.text or 'createElement("details")' in page.text
        assert "renderCaption" in page.text
        assert "#888" in page.text
        assert "11px" in page.text
        assert "source-item" in page.text
        assert "pre-wrap" in page.text
        assert "full_text" in page.text
        assert "/api/resume" not in page.text

        for approach, mode, check in (
            ("baseline", "mix", "id"),
            ("long", "mix", "group"),
            ("light", "hybrid", "relations"),
        ):
            response = client.post(
                "/api/chat",
                json={"message": "Кто умеет делать завивку?", "approach": approach, "mode": mode},
            )
            assert response.status_code == 200, response.text
            events = _sse_events(response.content)
            assert events, "empty SSE"
            assert "error" not in events[0], events
            payload = events[0]
            assert payload.get("content")
            assert "Ответ:" not in payload["content"]
            assert "[ID]" not in payload["content"]
            sources = payload.get("sources") or {}
            assert sources.get("used") is True
            assert sources.get("model")
            assert sources.get("label")
            assert payload.get("meta", {}).get("model") == sources["model"]
            assert [item["id"] for item in sources.get("items") or []] == ["R1"]
            assert "Кандидат №1" in sources["items"][0]["full_text"]
            assert len(sources["items"][0]["full_text"]) > len(sources["items"][0]["snippet"])
            assert "resumes" not in sources
            blob = json.dumps(sources, ensure_ascii=False)
            assert "R99" not in blob
            if check == "group":
                assert sources["items"][0].get("group_id") == "G0005"
            elif check == "relations":
                assert sources.get("mode") == "hybrid"
                assert sources["items"][0].get("relations")

        cook = client.post(
            "/api/chat",
            json={"message": "Кто больше всего подходит на роль повара?", "approach": "long", "mode": "mix"},
        )
        cook_events = _sse_events(cook.content)
        assert cook_events[0]["content"] != "[R367]"
        assert "[R367]" in cook_events[0]["content"]
        assert "Наиболее подходящий кандидат" in cook_events[0]["content"]
        assert cook_events[0]["sources"]["items"][0]["id"] == "R367"
        assert "Приготовление блюд" in cook_events[0]["sources"]["items"][0]["full_text"]

        for approach in ("baseline", "long", "light"):
            named = client.post(
                "/api/chat",
                json={"message": "Кандидат №1 чем занимается", "approach": approach, "mode": "mix"},
            )
            ev = _sse_events(named.content)[0]
            assert ev["sources"]["items"][0]["id"] == "R1", ev
            assert "Кандидат №1" in ev["sources"]["items"][0]["full_text"]
            first = client.post(
                "/api/chat",
                json={"message": "Чем занимается первый кандидат", "approach": approach, "mode": "mix"},
            )
            ev = _sse_events(first.content)[0]
            assert ev["sources"]["items"][0]["id"] == "R1", ev

        compared = client.post(
            "/api/chat",
            json={
                "message": "Сравни кандидатов №1 и №2: у кого больше опыта работы с людьми?",
                "approach": "long", "mode": "mix",
            },
        )
        ev = _sse_events(compared.content)[0]
        assert "[R1]" in ev["content"] and "[R2]" in ev["content"], ev
        assert [item["id"] for item in ev["sources"]["items"]] == ["R1", "R2"], ev
        assert ev["content"].count("..") == 0
        assert "головы. [R1]." not in ev["content"]

        multi = client.post(
            "/api/chat",
            json={
                "message": "Кого из двух разных кандидатов подобрать для столовой, если один будет готовить?",
                "approach": "baseline", "mode": "mix",
            },
        )
        ev = _sse_events(multi.content)[0]
        ids = [item["id"] for item in ev["sources"]["items"]]
        assert len(ids) >= 2, ev

        refusal = client.post(
            "/api/chat",
            json={"message": "Кто администрировал Kubernetes?", "approach": "baseline", "mode": "mix"},
        )
        events = _sse_events(refusal.content)
        assert events[0]["content"] == "В резюме нет достаточных сведений для ответа."
        refusal_sources = events[0].get("sources") or {}
        assert refusal_sources.get("used") is False
        assert refusal_sources.get("items") == []
        assert refusal_sources.get("label") == "Базовая RAG"
        assert refusal_sources.get("model")
        assert events[0].get("meta", {}).get("label") == "Базовая RAG"


def main_run():
    test_syntax()
    test_polish_and_citations()
    test_id_mapping_and_named_candidates()
    test_raw_resume_files()
    test_mock_engine()
    test_api_mock()
    print("all tests passed")


if __name__ == "__main__":
    main_run()
