"""Generate the managed Google Colab links block in README.md."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


SECTION_HEADING = "## Ссылки на актуальные Google Colab:"
BEGIN_MARKER = "<!-- BEGIN AUTO-GENERATED COLAB LINKS -->"
END_MARKER = "<!-- END AUTO-GENERATED COLAB LINKS -->"


class DuplicateWorkNumberError(ValueError):
    """Raised when two notebooks map to the same displayed work number."""


@dataclass(frozen=True)
class Work:
    number: int
    path: str


def discover_works(root: Path, directory: str, prefix: str) -> list[Work]:
    """Find matching notebooks and reject ambiguous work numbers."""
    source = root / directory
    if not source.exists():
        return []

    name_pattern = re.compile(
        rf"^{re.escape(prefix)}(?P<number>\d+)_(?P<suffix>.+)\.ipynb$"
    )
    works: list[Work] = []
    paths_by_number: dict[int, str] = {}

    for notebook in sorted(source.rglob("*.ipynb")):
        match = name_pattern.fullmatch(notebook.name)
        if match is None:
            continue

        number = int(match.group("number"))
        relative_path = notebook.relative_to(root).as_posix()
        previous = paths_by_number.get(number)
        if previous is not None:
            raise DuplicateWorkNumberError(
                f"Duplicate {prefix} number {number}: {previous}, {relative_path}"
            )

        paths_by_number[number] = relative_path
        works.append(Work(number=number, path=relative_path))

    return sorted(works, key=lambda work: work.number)


def colab_url(repository: str, branch: str, path: str) -> str:
    encoded_branch = quote(branch, safe="/")
    encoded_path = quote(path, safe="/")
    return (
        "https://colab.research.google.com/github/"
        f"{repository}/blob/{encoded_branch}/{encoded_path}"
    )


def render_group(
    title: str,
    empty_message: str,
    item_label: str,
    works: list[Work],
    repository: str,
    branch: str,
) -> list[str]:
    lines = [title]
    if not works:
        return [*lines, f"- {empty_message}"]

    for work in works:
        url = colab_url(repository, branch, work.path)
        lines.append(
            f"- {item_label} №{work.number}: "
            f"[Открыть в Google Colab]({url})"
        )
    return lines


def render_managed_block(
    labs: list[Work], practices: list[Work], repository: str, branch: str
) -> str:
    lines = [BEGIN_MARKER]
    lines.extend(
        render_group(
            "Лабораторные работы:",
            "Лабораторных работ пока нет.",
            "Лабораторная работа",
            labs,
            repository,
            branch,
        )
    )
    lines.append("")
    lines.extend(
        render_group(
            "Практические работы:",
            "Практических работ пока нет.",
            "Практическая работа",
            practices,
            repository,
            branch,
        )
    )
    lines.append(END_MARKER)
    return "\n".join(lines)


def replace_managed_block(readme: str, managed_block: str) -> str:
    begin_count = readme.count(BEGIN_MARKER)
    end_count = readme.count(END_MARKER)

    if begin_count == 1 and end_count == 1:
        begin = readme.index(BEGIN_MARKER)
        end = readme.index(END_MARKER) + len(END_MARKER)
        if begin >= end:
            raise ValueError("README Colab link markers are in the wrong order")
        return readme[:begin] + managed_block + readme[end:]

    if begin_count or end_count:
        raise ValueError("README must contain exactly one complete pair of Colab markers")

    heading_match = re.search(
        rf"(?m)^{re.escape(SECTION_HEADING)}[ \t]*$", readme
    )
    if heading_match is None:
        raise ValueError(f"README does not contain heading: {SECTION_HEADING}")

    body_start = heading_match.end()
    next_heading = re.search(r"(?m)^##\s+", readme[body_start:])
    body_end = (
        body_start + next_heading.start() if next_heading is not None else len(readme)
    )
    suffix = readme[body_end:].lstrip("\n")
    updated = readme[:body_start].rstrip() + "\n" + managed_block + "\n"
    if suffix:
        updated += "\n" + suffix
    return updated


def update_readme(root: Path, repository: str, branch: str) -> bool:
    """Update README and return True only when its content changed."""
    labs = discover_works(root, "labs", "Lab")
    practices = discover_works(root, "practices", "Practice")
    managed_block = render_managed_block(labs, practices, repository, branch)

    readme_path = root / "README.md"
    original = readme_path.read_text(encoding="utf-8")
    updated = replace_managed_block(original, managed_block)
    if updated == original:
        return False

    readme_path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub owner/repository (defaults to GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("GITHUB_REF_NAME", "main"),
        help="Branch used in generated links (defaults to GITHUB_REF_NAME or main).",
    )
    args = parser.parse_args(argv)
    if not args.repository:
        parser.error("--repository is required outside GitHub Actions")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        changed = update_readme(args.root.resolve(), args.repository, args.branch)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("README.md updated" if changed else "README.md is already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
