"""Run a small offline search demo on generated documents in a temporary directory."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import catalog
import encoder
import hybrid
import vectors


EXAMPLES = {
    "repair.md": "# Bicycle puncture repair\nRemove the wheel, find the hole in the inner tube and apply a patch.",
    "plants.md": "# Домашние растения\nЛетом проверяйте влажность земли перед поливом.",
    "budget.md": "# Monthly budget\nTrack grocery costs and recurring expenses.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="Использовать установленную E5")
    args = parser.parse_args()
    if args.semantic and not encoder.available():
        parser.error("Установи E5 и зависимости по README")
    with TemporaryDirectory() as tmp:
        sources = Path(tmp) / "examples"
        sources.mkdir()
        db = Path(tmp) / "demo.sqlite"
        for name, text in EXAMPLES.items():
            (sources / name).write_text(text, encoding="utf-8")
        catalog.refresh(db, [sources])
        if args.semantic:
            vectors.refresh(db)
        cases = [("влажность земли", "plants.md"), ("repair.md", "repair.md")]
        if args.semantic:
            cases.append(("чем заклеить проколотую велосипедную камеру", "repair.md"))
        for query, expected in cases:
            lexical = catalog.search(db, query)
            semantic = vectors.search(db, query) if args.semantic else []
            result = hybrid.combine(query, lexical, semantic)
            actual = Path(result[0]["path"]).name if result else None
            if actual != expected:
                raise SystemExit(f"{query!r}: ожидался {expected}, получен {actual}")
            print(f"{query} → {actual}")


if __name__ == "__main__":
    main()
