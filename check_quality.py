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
    "plants.md": "# Houseplants\nCheck the soil moisture before watering.",
    "budget.md": "# Monthly budget\nTrack grocery costs and recurring expenses.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="Use the installed E5 model")
    args = parser.parse_args()
    if args.semantic and not encoder.available():
        parser.error("Install E5 and its dependencies as described in the README")
    with TemporaryDirectory() as tmp:
        sources = Path(tmp) / "examples"
        sources.mkdir()
        db = Path(tmp) / "demo.sqlite"
        for name, text in EXAMPLES.items():
            (sources / name).write_text(text, encoding="utf-8")
        catalog.refresh(db, [sources])
        if args.semantic:
            vectors.refresh(db)
        cases = [("soil moisture", "plants.md"), ("repair.md", "repair.md")]
        if args.semantic:
            cases.append(("how to patch a flat bike tire", "repair.md"))
        for query, expected in cases:
            lexical = catalog.search(db, query)
            semantic = vectors.search(db, query) if args.semantic else []
            result = hybrid.combine(query, lexical, semantic)
            actual = Path(result[0]["path"]).name if result else None
            if actual != expected:
                raise SystemExit(f"{query!r}: expected {expected}, got {actual}")
            print(f"{query} → {actual}")


if __name__ == "__main__":
    main()
