"""Read and write labeled TSV files: a header row, then one text and label per line."""

from pathlib import Path

Row = tuple[str, str]


def read_tsv(path: Path) -> list[Row]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [tuple(line.split("\t")) for line in lines[1:]]


def write_tsv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["text\tlabel", *(f"{text}\t{label}" for text, label in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
