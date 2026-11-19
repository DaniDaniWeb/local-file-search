#!/usr/bin/python3
import datetime
import os
from pathlib import Path
import threading

import gi
gi.require_version("Gtk","4.0")
from gi.repository import Gio, GLib, Gtk, Gdk, Pango
import catalog
import encoder
import qwen
import hybrid
import vectors


def later(fn,*args):
    def invoke():
        fn(*args)
        return False
    GLib.idle_add(invoke)


def background(fn):
    threading.Thread(target=fn,daemon=True).start()


def open_file(path):
    file = Path(path)
    if not file.is_file():
        raise ValueError("Файл перемещён или удалён. Обнови индекс.")
    return Gio.AppInfo.launch_default_for_uri(file.as_uri(), None)


class SearchApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id=catalog.APP_ID)
        self.window = None
        self.serial = 0
        self.model_lock = threading.Lock()
        self.vector_lock = threading.Lock()
        self.query = ""
        self.open_on_enter = False
        self.translations = {}
        self.direct_rows = []
        self.expanded_rows = []
        self.translated_semantic = []
        self.expansion_status = ""
        self.semantic_rows = []
        self.translated_rows = []
        self.vector_status = ""
        self.qwen_status = ""

    def do_activate(self):
        if not self.window:
            catalog.initialize(catalog.DB)
            self.build()
            try:
                catalog.config()
            except (ValueError, OSError) as exc:
                self.note.set_text("Настройки: " + str(exc))
        self.window.present()
        self.open_on_enter = False
        self.entry.grab_focus()
        self.entry.select_region(0,-1)

    def build(self):
        self.window = Gtk.ApplicationWindow(application=self,title="Поиск файлов")
        self.window.set_default_size(680,475)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=9)
        for side in ("top","bottom","start","end"):
            getattr(box,"set_margin_"+side)(14)
        self.window.set_child(box)
        self.window.connect("close-request",self.hide_window)
        self.entry = Gtk.Entry(placeholder_text="Какой файл найти?",max_length=2000)
        self.entry.connect("activate",self.submit)
        box.append(self.entry)
        self.note = Gtk.Label(label="Введи описание файла и нажми Enter",xalign=0,wrap=True)
        self.note.set_lines(2)
        self.note.set_ellipsize(Pango.EllipsizeMode.END)
        box.append(self.note)
        self.direct_only = Gtk.ToggleButton(label="Только исходные слова")
        self.direct_only.set_halign(Gtk.Align.START)
        self.direct_only.connect("toggled", self.change_view)
        self.direct_only.set_visible(False)
        box.append(self.direct_only)
        self.list = Gtk.ListBox(activate_on_single_click=True)
        self.list.connect("row-activated",self.open_row)
        scroll = Gtk.ScrolledWindow(vexpand=True)
        scroll.set_child(self.list)
        box.append(scroll)
        self.index_note = Gtk.Label(xalign=0,wrap=True)
        with catalog.connect(catalog.DB) as con:
            count = con.execute("SELECT count(*) FROM files").fetchone()[0]
        try:
            indexed, _ = vectors.status(catalog.DB)
        except (ImportError, RuntimeError):
            indexed = 0
        if count:
            message = f"Файлов: {count}"
            if encoder.available():
                message += f" · смысловой индекс: {indexed}"
            if encoder.available() and indexed < count:
                message += " · обнови индекс для полного охвата"
        else:
            message = "Для первого поиска обнови индекс"
        self.index_note.set_text(message)
        box.append(self.index_note)
        buttons = Gtk.Box(spacing=8)
        box.append(buttons)
        self.update = Gtk.Button(label="Обновить индекс")
        self.update.connect("clicked",self.refresh)
        buttons.append(self.update)
        settings = Gtk.Button(label="Настройки")
        settings.connect("clicked",self.settings)
        buttons.append(settings)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed",self.key)
        self.window.add_controller(keys)

    def hide_window(self,*_):
        self.window.set_visible(False)
        return True

    def settings(self,*_):
        try:
            editor = Gio.AppInfo.get_default_for_type("text/plain",False)
            if not editor:
                raise ValueError("Установи текстовый редактор для настроек")
            editor.launch([Gio.File.new_for_path(str(catalog.CONFIG))],None)
        except Exception as exc:
            self.note.set_text(str(exc))

    def key(self,controller,key,code,mods):
        if key == Gdk.KEY_Escape:
            self.window.set_visible(False)
            return True
        if key == Gdk.KEY_Down and (self.window.get_focus() == self.entry or (self.window.get_focus() and self.window.get_focus().is_ancestor(self.entry))) and self.list.get_row_at_index(0):
            self.list.get_row_at_index(0).grab_focus()
            return True
        return False

    def refresh(self,*_):
        self.update.set_sensitive(False)
        self.index_note.set_text("Обход папок…")
        def work():
            try:
                progress = lambda s: later(self.index_note.set_text,s)
                msg = catalog.refresh(catalog.DB,catalog.config()["folders"],progress)
                try:
                    if encoder.available():
                        msg += " · " + vectors.refresh(catalog.DB,progress)
                except Exception as exc:
                    msg += " · Индекс по смыслу: " + str(exc)
            except Exception as exc:
                msg = str(exc)
            later(self.index_note.set_text,msg)
            later(self.update.set_sensitive,True)
        background(work)

    def submit(self,*_):
        query = self.entry.get_text().strip()
        if not query:
            return
        if self.open_on_enter and query == self.query and self.list.get_selected_row():
            self.open_row(self.list,self.list.get_selected_row())
            return
        self.serial += 1
        serial = self.serial
        self.query = query
        self.open_on_enter = True
        self.direct_rows = []
        self.expanded_rows = []
        self.translated_semantic = []
        self.expansion_status = ""
        self.semantic_rows = []
        self.translated_rows = []
        self.vector_status = ""
        self.qwen_status = ""
        self.direct_only.set_visible(False)
        self.direct_only.set_active(False)
        self.list.remove_all()
        self.note.set_tooltip_text(None)
        self.note.set_text("Поиск по индексу…")
        def publish(fn,*args):
            later(lambda: fn(*args) if serial == self.serial else None)
        def work():
            try:
                rows = catalog.search(catalog.DB,query)
                if rows and Path(rows[0]["path"]).name.casefold() == query.casefold():
                    publish(self.render,rows)
                    publish(self.note.set_text,"Точное имя файла")
                    return
                publish(self.show_direct, rows)
                if not catalog.tokens(query):
                    publish(self.note.set_text,"По формату файла")
                    return
                if encoder.available():
                    background(semantic_work)
                options = catalog.config().get("qwen")
                if options is None:
                    return
                model = options["model"]
                publish(self.model_state,"Ожидание перевода · " + model)
                with self.model_lock:
                    if serial != self.serial:
                        return
                    key = (options["base_url"],model,query)
                    if key not in self.translations:
                        terms,kind = qwen.expand(query,options,lambda s:publish(self.model_state,s))
                        if len(self.translations) >= 64:
                            self.translations.pop(next(iter(self.translations)))
                        self.translations[key] = terms,kind
                    terms,kind = self.translations[key]
                if serial != self.serial:
                    return
                expanded = catalog.search(catalog.DB,query,terms,kind) if terms else rows
                publish(self.show_expanded,expanded,terms,model)
                description = hybrid.translated_description(query,terms)
                if description and encoder.available() and serial == self.serial:
                    try:
                        with self.vector_lock:
                            if serial != self.serial:
                                return
                            other = vectors.search(catalog.DB,query,description=description)
                        publish(self.show_translated_semantic,other)
                    except Exception as exc:
                        publish(self.show_translated_semantic,[],str(exc))
            except Exception as exc:
                publish(self.model_state,str(exc))
        def semantic_work():
            try:
                with self.vector_lock:
                    if serial != self.serial:
                        return
                    found = vectors.search(catalog.DB, query)
                publish(self.show_semantic,found)
            except Exception as exc:
                publish(self.show_semantic,[],"Поиск по смыслу: "+str(exc))
        background(work)

    def model_state(self, status):
        self.qwen_status = status
        self.update_note()

    def show_semantic(self, rows, status=None):
        self.semantic_rows = rows
        self.vector_status = status or ("Поиск по смыслу" if rows else "")
        self.update_views()

    def show_direct(self, rows):
        self.direct_rows = rows
        self.translated_rows = rows
        self.qwen_status = ""
        self.render(rows)
        self.note.set_text("По исходным словам")

    def show_expanded(self, rows, terms, model):
        self.translated_rows = rows
        pairs = [f"{word} → {', '.join(aliases)}" for word,aliases in terms.items()]
        self.qwen_status = model + (" · " + " · ".join(pairs) if pairs else " · по исходным словам")
        self.update_views()

    def show_translated_semantic(self, rows, error=""):
        self.translated_semantic = rows
        if error:
            self.vector_status = "Поиск по смыслу: " + error
        self.update_views()

    def update_views(self):
        self.expanded_rows = hybrid.combine(self.query,self.translated_rows,self.semantic_rows,self.translated_semantic)
        self.expansion_status = " · ".join(filter(None,(self.vector_status,self.qwen_status)))
        def visible_results(items):
            return [(item["path"], item["weak"], item["excerpt"]) for item in items]
        self.direct_only.set_visible(visible_results(self.direct_rows) != visible_results(self.expanded_rows))
        self.change_view()

    def change_view(self, *_):
        if not self.expansion_status:
            return
        original = self.direct_only.get_active()
        rows = self.direct_rows if original else self.expanded_rows
        self.render(rows)
        self.update_note()

    def update_note(self):
        original = self.direct_only.get_active()
        rows = self.direct_rows if original else self.expanded_rows
        status = (" · ".join(filter(None,(self.vector_status,self.qwen_status))) if not original else "") or "По исходным словам"
        if rows and all(row["weak"] for row in rows) and not any(row.get("semantic") for row in rows):
            status = "Только частичные совпадения · " + status
        self.note.set_text(status)
        self.note.set_tooltip_text(status)

    def render(self,rows):
        selected = self.list.get_selected_row()
        path = selected.path if selected else None
        focused = self.window.get_focus()
        keep_focus = focused and focused.is_ancestor(self.list)
        self.list.remove_all()
        self.list.set_placeholder(Gtk.Label(label="В индексе нет совпадений по этому запросу"))
        chosen = None
        for result in rows:
            row = Gtk.ListBoxRow()
            row.path = result["path"]
            info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=2)
            info.set_margin_top(4)
            info.set_margin_bottom(4)
            row.set_child(info)
            date = datetime.datetime.fromtimestamp(result["mtime"]/1e9).strftime("%d.%m.%Y %H:%M")
            lines = [(Path(row.path).name,"heading"),(row.path+"  ·  "+date,"dim-label")]
            if result["excerpt"]:
                lines.append((" ".join(result["excerpt"].split()),"caption"))
            if result.get("semantic"):
                lines.append(("Похожий по смыслу фрагмент","dim-label"))
            elif result["weak"]:
                reason = "Неуверенное совпадение"
                missing = result.get("missing_anchors", ())
                if missing:
                    reason += " · нет подтверждения для " + ", ".join(f"«{w}»" for w in missing)
                lines.append((reason,"warning"))
            for value,style in lines:
                label = Gtk.Label(label=value,xalign=0,ellipsize=Pango.EllipsizeMode.END)
                label.set_tooltip_text(value)
                label.add_css_class(style)
                info.append(label)
            self.list.append(row)
            if keep_focus and row.path == path:
                chosen = row
        chosen = chosen or self.list.get_row_at_index(0)
        if chosen:
            self.list.select_row(chosen)
            if keep_focus:
                chosen.grab_focus()

    def open_row(self,widget,row):
        try:
            open_file(row.path)
            self.window.set_visible(False)
        except Exception as exc:
            self.note.set_text(str(exc))


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(SearchApp().run(None))
