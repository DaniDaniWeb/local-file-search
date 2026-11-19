from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import catalog
import hybrid
import vectors


class TestVectors(unittest.TestCase):
    def test_cache_refresh_removal_and_file_filter(self):
        batches = []

        class Encoder:
            def split(self, body, **kwargs):
                yield body

            def encode(self, texts, *, query):
                if not query:
                    batches.extend(texts)
                return np.ones((len(texts), 3))

        with tempfile.TemporaryDirectory() as folder, patch.object(vectors, "_model", Encoder()):
            temp = Path(folder)
            root = temp / "docs"
            root.mkdir()
            db = temp / "catalog.sqlite"
            one, two = root / "one.md", root / "two.md"
            for file in (one, two):
                file.write_text("Shared passage")
            catalog.refresh(db, [root])
            vectors.refresh(db)
            self.assertEqual(batches.count("Shared passage"), 1)
            self.assertEqual(vectors.status(db)[0], 2)
            batches.clear()
            vectors.refresh(db)
            self.assertEqual(batches, [])

            one.write_text("Updated passage")
            two.unlink()
            catalog.refresh(db, [root])
            self.assertEqual(vectors.search(db, "meaning"), [])
            vectors.refresh(db)
            self.assertEqual(batches, ["Updated passage"])
            self.assertEqual(vectors.search(db, "PDF about meaning"), [])
            rows = vectors.search(db, "meaning")
            self.assertEqual([r["path"] for r in rows], [str(one)])

    def test_literal_result_and_named_entity(self):
        original = dict(path="/local/book.pdf", size=100, score=200, weak=False,
                        kind="pdf", excerpt="Sigmund Freud")
        adjacent = dict(path="/local/other.pdf", size=80, score=.86, weak=True,
                        kind="pdf", excerpt="Carl Jung and analytical psychology")
        image = dict(adjacent, path="/local/frame.png",kind="png")
        self.assertEqual(hybrid.combine("PDF про Freud", [original], [adjacent,image]), [original])

    def test_ocr_phrase_with_insertion(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "index.sqlite"
            catalog.initialize(db)
            with catalog.connect(db) as con:
                for name, text in [
                    ("01.png","теория цветового круга. " + "Пример. "*18 + "Бессознательного у Фрейда"),
                    ("02.png","ТЕОРИЯ БЕССОЗНАТЕЛЬНОГО ЗИГМУНДА ФРЕЙЛДА"),
                ]:
                    ident = con.execute("INSERT INTO files(path,kind,mtime,size,ctime,pending) VALUES(?,?,1,1,1,0)",
                                        (str(Path(folder)/name),"png")).lastrowid
                    con.execute("INSERT INTO docs(rowid,name,body) VALUES(?,?,?)",(ident,name,text))
            first = catalog.search(db,"Теория бессознательного Фрейда")[0]
            self.assertEqual(Path(first["path"]).name,"02.png")
            self.assertFalse(first["weak"])
            self.assertIn("ФРЕЙЛДА",first["excerpt"])
