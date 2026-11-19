#!/usr/bin/env python3
"""Register the launcher and a free shortcut in a GNOME session."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gio, Gtk

from catalog import APP_ID


SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
PREFIX = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"
LABEL = "Локальный поиск файлов"


def desktop_arg(value):
    escaped = re.sub(r'([\\`"$])', r'\\\1', value).replace("\\", "\\\\").replace("%", "%%")
    return '"' + escaped + '"'


def install_shortcut(argv):
    if Gio.SettingsSchemaSource.get_default().lookup(SCHEMA, True) is None:
        return None
    manager = Gio.Settings.new(SCHEMA)
    paths = manager.get_strv("custom-keybindings")
    path = PREFIX + "local-file-search-public/"
    entry = Gio.Settings.new_with_path(SCHEMA + ".custom-keybinding", path)
    shortcut = entry.get_string("binding") if path in paths else ""
    if not shortcut:
        values = subprocess.check_output(["gsettings", "list-recursively"], text=True)
        assigned = re.findall(r"'([^']+)'", values)
        for existing in paths:
            assigned.append(Gio.Settings.new_with_path(SCHEMA + ".custom-keybinding", existing).get_string("binding"))

        def key(value):
            valid, code, modifiers = Gtk.accelerator_parse(value)
            return (code, int(modifiers)) if valid else None

        used = {key(v) for v in assigned if "<" in v}
        choices = ("<Control><Alt>space", "<Super><Shift>f", "<Control><Alt>F8")
        shortcut = next((v for v in choices if key(v) not in used), None)
        if shortcut is None:
            return None
    entry.set_string("name", LABEL)
    entry.set_string("command", shlex.join(argv))
    entry.set_string("binding", shortcut)
    if path not in paths:
        manager.set_strv("custom-keybindings", [*paths, path])
    Gio.Settings.sync()
    return shortcut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-shortcut", action="store_true", help="Установить только ярлык")
    options = parser.parse_args()
    app = Path(__file__).resolve().with_name("app.py")
    args = [sys.executable, str(app)]
    apps = Path(os.environ.get("XDG_DATA_HOME", Path.home()/".local/share")) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / (APP_ID + ".desktop")).write_text(
        "[Desktop Entry]\nType=Application\n"
        f"Name={LABEL}\n"
        "Comment=Поиск документов и текста на изображениях\nIcon=system-search\n"
        f"Exec={' '.join(desktop_arg(arg) for arg in args)}\n"
        "Terminal=false\nCategories=Utility;GTK;\nStartupNotify=true\n", encoding="utf-8")
    shortcut = install_shortcut(args) if not options.no_shortcut else None
    print(LABEL + (" · " + shortcut if shortcut else ""))


if __name__ == "__main__":
    main()
