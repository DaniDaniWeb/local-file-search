import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cairo
from PIL import Image, ImageDraw, ImageFont
from gi.repository import GLib
import app
import catalog as c


class Scenarios(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)/"files"
        self.root.mkdir()
        self.db = self.root.parent/"index.sqlite"
        self.pdf = self.root/"01.pdf"
        surface = cairo.PDFSurface(str(self.pdf),620,100)
        ctx = cairo.Context(surface)
        ctx.set_font_size(18)
        ctx.move_to(12,45)
        ctx.show_text("Математика для машинного обучения")
        surface.finish()
        self.note = self.root/"02.md"
        self.note.write_text("Конспект о рекрутинге в HFT")
        font = ImageFont.truetype("DejaVuSans.ttf",30)
        image = Image.new("RGB",(760,100),"white")
        ImageDraw.Draw(image).text((14,20),"Minecraft crash log: fatal exception",font=font,fill="black")
        self.pic = self.root/"03.png"
        image.save(self.pic)
        c.refresh(self.db,[self.root])

    def test_search_and_open(self):
        for query,file in [("Найди PDF учебника по математике для машинного обучения",self.pdf),
                           ("Найди конспект по рекрутингу в HFT", self.note),
                           ("Найди скриншот с логами краша Minecraft",self.pic)]:
            row = c.search(self.db,query)[0]
            self.assertEqual(row["path"],str(file))
            self.assertTrue(row["excerpt"])
            with patch.object(app.Gio.AppInfo,"launch_default_for_uri",return_value=True) as launch:
                self.assertTrue(app.open_file(row["path"]))
                launch.assert_called_once_with(file.as_uri(),None)

    def test_incremental(self):
        self.note.write_text("Архитектура процессора")
        with patch.object(c,"extract",wraps=c.extract) as extract:
            c.refresh(self.db,[self.root])
            extract.assert_called_once_with(self.note,"md")
        self.assertEqual(c.search(self.db,"процессор")[0]["path"],str(self.note))
        self.assertFalse(c.search(self.db,"HFT"))

    def test_cold_model(self):
        from unittest.mock import Mock
        ui = app.SearchApp()
        ui.entry = Mock()
        ui.entry.get_text.return_value = "HFT"
        ui.list,ui.note,ui.direct_only = Mock(),Mock(),Mock()
        ui.direct_only.get_active.return_value = False
        released = threading.Event()
        ordinary = threading.Event()
        enhanced = threading.Event()
        def render(rows):
            self.assertEqual(rows[0]["path"],str(self.note))
            (enhanced if released.is_set() else ordinary).set()
        ui.render = render
        def slow(*args):
            released.wait(5)
            return {"HFT":["high frequency trading"]},""
        with patch.object(c,"DB",self.db),patch.object(c,"config",return_value=dict(c.DEFAULT,qwen={"model":"fixture","base_url":"http://127.0.0.1:11434/v1"})),patch.object(app.qwen,"expand",side_effect=slow):
            ui.submit()
            import time
            loop = GLib.MainContext.default()
            def wait_for(event):
                end = time.monotonic()+3
                while not event.is_set() and time.monotonic()<end:
                    loop.iteration(False)
                    time.sleep(.005)
                self.assertTrue(event.is_set())
            try:
                wait_for(ordinary)
            finally:
                released.set()
            wait_for(enhanced)

    def test_format_and_missing_screenshot(self):
        self.pic.unlink()
        (self.root/"MinecraftInstaller.exe").write_bytes(b"")
        (self.root/"Minecraft crash log.md").write_text("Minecraft fatal crash")
        c.refresh(self.db,[self.root])
        hits = c.search(self.db,"скриншот с логами краша Minecraft",{"краша":["crash"]})
        self.assertEqual(hits,[])

    def test_partial_ocr_marks_missing_name(self):
        image = Image.new("RGB",(760,100),"white")
        ImageDraw.Draw(image).text((14,20),"Database crash detected",
                                   font=ImageFont.truetype("DejaVuSans.ttf",30),fill="black")
        image.save(self.pic)
        c.refresh(self.db,[self.root])
        rows = c.search(self.db,"скриншот краша Minecraft",{"краша":["crash"]})
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]["weak"])
        self.assertEqual(rows[0]["missing_anchors"],["Minecraft"])

    def test_translation_retrieves_english_note(self):
        manual = self.root/"reference.md"
        manual.write_text("# Fault tolerance\nA handbook for system designers.")
        c.refresh(self.db,[self.root])
        query = "заметка об отказоустойчивости"
        direct = c.search(self.db,query)
        self.assertNotIn(str(manual), [row["path"] for row in direct])
        expanded = c.search(self.db,query,{"отказоустойчивости":["fault tolerance"]})
        self.assertEqual(expanded[0]["path"],str(manual))
        self.assertIn("fault tolerance",expanded[0]["excerpt"].lower())

    def test_direct_hit_survives_noisy_model(self):
        note = self.root/"защита памяти.md"
        note.write_text("# Защита памяти\nПрава доступа к страницам.")
        distractor = self.root/"data science.md"
        distractor.write_text("# Data science\n" + "data science "*50)
        c.refresh(self.db,[self.root])
        hints = {"защита памяти":["data science"]}
        results = c.search(self.db,"защита памяти",hints,"pdf")
        self.assertEqual(results[0]["path"],str(note))

    def test_model_output_is_grounded(self):
        reply = '{"смежная тема":["neural networks"],"формат":["pdf"],"ошибок":["errors"]}'
        self.assertEqual(app.qwen.aliases_from_reply(reply,"шаблон журнала ошибок"),{"ошибок":["errors"]})


class PublicSettings(unittest.TestCase):
    def test_defaults_disable_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(c, "CONFIG", Path(directory) / "config.json"):
                self.assertIsNone(c.config()["qwen"])

    def test_qwen_rejects_remote_and_redirects(self):
        for address in ("http://example.org/v1", "http://127.0.0.1@evil.example/v1"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                app.qwen.settings({"base_url": address, "model": "test"})
        options = {"base_url": "http://localhost:11434/v1", "model": "test"}
        with patch.object(app.qwen.http.client, "HTTPConnection") as client:
            conn = client.return_value
            conn.getresponse.return_value.status = 302
            with self.assertRaisesRegex(ValueError, "HTTP 302"):
                app.qwen.expand("журнал ошибок", options, lambda _: None)
            client.assert_called_once_with("127.0.0.1", 11434, timeout=12)
            self.assertEqual(conn.request.call_args.args[:2], ("POST", "/v1/chat/completions"))
            conn.close.assert_called_once()
