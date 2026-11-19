"""Local incremental file catalog and ranked full-text search."""
import contextlib
import difflib
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile

HOME = Path.home()
APP_DIR = "local-file-search-public"
APP_ID = "org.localfilesearch.Desktop"
CONFIG = Path(os.getenv("XDG_CONFIG_HOME", HOME / ".config")) / APP_DIR / "config.json"
DB = Path(os.getenv("XDG_DATA_HOME", HOME / ".local/share")) / APP_DIR / "index.sqlite"
DEFAULT_FOLDERS = [HOME / name for name in ("Downloads", "Documents", "Pictures/Screenshots")]
DEFAULT = {"folders": [str(path) for path in DEFAULT_FOLDERS if path.is_dir()], "qwen": None}
SKIP = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", ".cache",
        ".tox", ".mypy_cache", ".pytest_cache", ".next", ".ruff_cache", ".gradle"}
IMAGES = {"png", "jpeg", "jpg"}
STOP = set("найди найти покажи где файл файлы мне мой мои моя с со к по для о на в из и или это про пожалуйста the of a for and скриншот скриншоты pdf png jpeg jpg markdown md".split())
WORD = re.compile(r"[^\W_]+", re.UNICODE)
SEARCH_REQUEST = re.compile(
    r"^(?:найди(?:те)?|найти|покажи(?:те)?|ищу|find(?: me)?|show(?: me)?|look for|search for)\s+",
    re.I,
)


def config():
    if not CONFIG.exists():
        CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with CONFIG.open("x", encoding="utf-8") as out:
            json.dump(DEFAULT, out, ensure_ascii=False, indent=2)
    data = json.loads(CONFIG.read_text())
    if not isinstance(data, dict):
        raise ValueError("Settings must be a JSON object")
    if not isinstance(data.get("folders"), list) or not all(isinstance(p, str) and Path(p).expanduser().is_absolute() for p in data["folders"]):
        raise ValueError("The folders setting must list absolute paths or paths starting with ~/")
    if data.get("qwen") is not None:
        from qwen import settings
        settings(data["qwen"])
    return data


@contextlib.contextmanager
def connect(db):
    con = sqlite3.connect(db, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def initialize(db):
    db.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with connect(db) as con:
        con.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT UNIQUE,kind TEXT,
            mtime INTEGER,size INTEGER,ctime INTEGER,pending INTEGER);
        CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(name,body, prefix='3 4');
        CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(docs,'row');
        """)


def extract(path, kind):
    if kind in ("md", "markdown"):
        with path.open("rb") as src:
            return src.read(4_000_000).decode("utf-8-sig", "replace")
    args = (["pdftotext", "-enc", "UTF-8", str(path), "-"] if kind == "pdf" else
            ["tesseract", str(path), "stdout", "-l", "rus+eng", "--psm", "11"])
    with tempfile.TemporaryFile() as out:
        subprocess.run(args, stdout=out, stderr=subprocess.DEVNULL, check=True, timeout=90,
                       env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        out.seek(0)
        return out.read(4_000_000).decode("utf-8", "replace")


def refresh(db, folders, progress=lambda msg: None):
    initialize(db)
    with db.with_suffix(".lock").open("a") as lock, connect(db) as con:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("An index update is already running") from None
        seen, missed = set(), []
        queue = []
        known = {r["path"]: r for r in con.execute("SELECT * FROM files")}
        def failed(error):
            missed.append(Path(error.filename))
        for folder in folders:
            for directory, dirs, names in os.walk(Path(folder).expanduser(), onerror=failed):
                dirs[:] = [d for d in dirs if d not in SKIP]
                for name in names:
                    path = Path(directory, name)
                    key = str(path)
                    if key in seen:
                        continue
                    try:
                        info = path.lstat()
                    except OSError:
                        missed.append(path)
                        continue
                    if not stat.S_ISREG(info.st_mode) or os.fsencode(key).decode("utf8","ignore") != key:
                        continue
                    seen.add(key)
                    stamp = info.st_mtime_ns, info.st_size, info.st_ctime_ns
                    old = known.get(key)
                    kind = path.suffix[1:].lower()
                    pending = kind in IMAGES | {"md", "markdown", "pdf"}
                    if old and stamp == (old["mtime"], old["size"], old["ctime"]):
                        pending = old["pending"]
                        ident = old["id"]
                    else:
                        ident = con.execute("""INSERT INTO files(path,kind,mtime,size,ctime,pending)
                            VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET kind=excluded.kind,
                            mtime=excluded.mtime,size=excluded.size,ctime=excluded.ctime,pending=excluded.pending
                            RETURNING id""", (key,kind,*stamp,pending)).fetchone()[0]
                        con.execute("DELETE FROM docs WHERE rowid=?", (ident,))
                        con.execute("INSERT INTO docs(rowid,name,body) VALUES(?,?, '')", (ident,name))
                    if pending:
                        priority = 0 if kind not in IMAGES else (1 if re.search("screenshot|скрин|снимок", key, re.I) else 2)
                        queue.append((priority,path,kind,ident))
        for key in known.keys() - seen:
            if any(p == Path(key) or p in Path(key).parents for p in missed):
                continue
            ident = known[key]["id"]
            con.execute("DELETE FROM docs WHERE rowid=?", (ident,))
            con.execute("DELETE FROM files WHERE id=?", (ident,))
        con.commit()
        errors = 0
        for i, (_,path,kind,ident) in enumerate(sorted(queue)):
            progress(f"Extracting text · {i+1} из {len(queue)}")
            try:
                body = extract(path, kind)
            except (OSError, subprocess.SubprocessError):
                errors += 1
                continue
            with con:
                con.execute("UPDATE docs SET body=? WHERE rowid=?", (body,ident))
                con.execute("UPDATE files SET pending=0 WHERE id=?", (ident,))
        return f"{len(seen)} files · Text extracted from {len(queue)-errors} · {errors} extraction errors · {len(missed)} unavailable paths"


# Format words are constraints. Genre words still carry evidence (especially
# in a title), but say little about the subject of a document.
IMAGE_WORD = re.compile(
    r"(?:скриншот|скрин|изображени|картинк|фотографи|screenshot|image|photo|picture)"
    r"(?:а|и|у|ы|е|я|ю|й|s)?$"
)
FILLER = STOP | set("""
    ищу искать ищи отыщи покажите нужен нужна нужно нужны этот эту эти мой мою
    где был была были там который которое которой котором которых которому
    помню написано написан надпись надписью надписями словом словами содержит
    можешь каком какие каким содержащий содержащая есть знаю забыл именно
    назывался называется при у об чтобы формата формате во экрана
    about an at by can could contain contains containing file files find
    for from i in is it look looking me my named need of on please says search
    show that the these this those to want which with written you would
""".split())
LOG_FORMS = {"лог","логи","лога","логов","логом","логам","логами","логах"}
GENRE = {"log", "note", "notes", "document", "book", "textbook", "проект", "метод", "способ", "подход", "конспект", "учебник", "заметк", "лог", "текст", "книг", "документ", "запис", "материал"}
ENDING = re.compile(r"(иями|ами|ями|ыми|ими|ого|его|ому|ему|иях|ях|ах|ям|ам|ом|ем|ой|ей|ов|ые|ый|ий|ая|яя|ую|юю|ое|ие|ы|и|а|я|у|ю|е|о)$")
MARKDOWN = {"md", "markdown"}
FORMAT = {"pdf","pdfs","пдф","пдфку","маркадаун","маркдаун","фото","снимок","screenshot"} | IMAGES | MARKDOWN


def words(text):
    return WORD.findall(text.casefold().replace("ё", "е"))


def stem(word):
    if word == "logs":
        return "log"
    if len(word) >= 5 and re.fullmatch("[а-я]+", word):
        word = ENDING.sub("", word)
        # Productive borrowed nouns such as рекрутинг / рекрутер.
        if word.endswith("инг") and len(word) >= 8:
            return word[:-3]
    return word


def tokens(query):
    return list(dict.fromkeys(stem(w) for w in words(query)
                              if w not in FILLER | FORMAT and not IMAGE_WORD.fullmatch(w)))[:18]


def requested_types(query):
    vocabulary = set(words(query))
    kinds = vocabulary & (IMAGES | {"pdf"})
    if "jpeg" in kinds:
        kinds.add("jpg")
    if vocabulary & {"pdfs","пдф","пдфку"}:
        kinds.add("pdf")
    if vocabulary & {"md","markdown","маркдаун","маркадаун"}:
        kinds.update(MARKDOWN)
    if not kinds and (vocabulary & {"фото","снимок"} or any(IMAGE_WORD.fullmatch(w) for w in vocabulary)):
        kinds.update(IMAGES)
    return kinds


@dataclass(frozen=True)
class Form:
    # A translated phrase is one indivisible match. It satisfies only the
    # source concepts it translated, even if many equivalent forms are found.
    phrase: tuple
    groups: tuple
    translated: bool = False
    fuzzy: bool = False

    def fts(self):
        if len(self.phrase) > 1:
            # Allow a qualifier between translated words, for instance
            # "procedural content generation". Check order in matches().
            terms = " ".join('"'+w+'"' for w in self.phrase)
            return f"NEAR({terms}, {len(self.phrase)})"
        w = self.phrase[0]
        variant = [w]
        if "е" in w:
            variant += [w[:i]+"ё"+w[i+1:] for i,c in enumerate(w) if c=="е"][:2]
        if w == "лог" and not self.translated:
            variant = sorted(LOG_FORMS)
        if w == "log":
            variant = ["log", "logs"]
        prefix = "*" if len(w) >= 4 else ""
        return "("+" OR ".join('"'+v+'"'+prefix for v in variant)+")"

    def matches(self, passage):
        if len(self.phrase) == 1:
            w = self.phrase[0]
            if w == "лог" and not self.translated:
                return not LOG_FORMS.isdisjoint(passage)
            if w == "log":
                return "log" in passage or "logs" in passage
            if len(w) >= 4:
                return any(token.startswith(w) for token in passage)
            return w in passage
        n = len(self.phrase)
        for i, token in enumerate(passage):
            if token != self.phrase[0]:
                continue
            next_word = 1
            for candidate in passage[i+1:i+n+2]:
                if candidate == self.phrase[next_word]:
                    next_word += 1
                    if next_word == n:
                        return True
        return False


def aliases_for(hints, original):
    if not isinstance(hints, dict):
        # Older clients supplied an unlabelled list. A single query concept
        # is the only case where its translation is unambiguous.
        hints = {original[0]:hints} if len(original)==1 and isinstance(hints,(list,tuple)) else {}
    forms = []
    for source, equivalents in list(hints.items())[:10]:
        if not isinstance(source,str) or not isinstance(equivalents,(list,tuple)):
            continue
        groups = tuple(tokens(source))
        if not groups or not set(groups).issubset(original):
            continue
        for equivalent in equivalents[:2]:
            if isinstance(equivalent,str):
                phrase = tuple(words(equivalent[:100]))
                if 0 < len(phrase) <= 6 and phrase != groups:
                    forms.append(Form(phrase, groups, translated=True))
    return list(dict.fromkeys(forms))


def header(kind, text):
    if kind not in MARKDOWN:
        return " ".join(text.split()[:42])
    title = re.search(r"(?m)^#\s+(.+)$",text)
    front = re.match(r"\A---\s*\n(.*?)\n---",text,re.S)
    parts = [title[1]] if title else []
    if front:
        for field in re.finditer(r"(?m)^(?:title|tags|aliases):[ \t]*(.*(?:\n[ \t]+-[^\n]*)*)",front[1]):
            parts.append(field[1])
    # The opening paragraph of a note often explains an opaque title.
    introduction = text[front.end():] if front else text
    parts.append(" ".join(introduction.split()[:40]))
    return " ".join(parts)


def evidence(forms, passage):
    found = {}
    for form in forms:
        if form.matches(passage):
            strength = 0.55 if form.fuzzy else (0.93 if form.translated else 1.0)
            for group in form.groups:
                found[group] = max(found.get(group,0),strength)
    return found


def one_edit(left, right):
    """One OCR insertion, deletion or substitution inside a sufficiently long word."""
    if min(len(left),len(right)) < 5 or left[0] != right[0] or abs(len(left)-len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a,b in zip(left,right)) == 1
    short, long = sorted((left,right),key=len)
    first = next((i for i,char in enumerate(short) if char != long[i]),len(short))
    return short[first:] == long[first+1:]


def phrase_in_text(concepts, text, ocr):
    """Find all original concepts in order with at most three intervening words.

    Permit one single-character error in OCR only when at least two other
    concepts independently confirm the phrase. Keep the original excerpt.
    """
    if not 2 <= len(concepts) <= 8:
        return None
    spans = list(WORD.finditer(text))
    passage = [m.group().casefold().replace("ё","е") for m in spans]
    # Each state represents a possible start and progress through the query.
    # Bound the state lifetime by the length of the query plus the permitted gap.
    states = []
    best = None
    for i,word in enumerate(passage):
        normalized = stem(word)
        active = []
        for start,matched,corrected in [*states,(i,0,False)]:
            gap = i - start - matched
            if gap > 3:
                continue
            if matched:
                active.append((start,matched,corrected))
            concept = concepts[matched]
            literal = word == concept or normalized == concept
            changed = False
            if not literal:
                if not ocr or corrected or len(concepts) < 3 or not one_edit(concept,normalized):
                    continue
                changed = True
            altered = corrected or changed
            if matched + 1 < len(concepts):
                active.append((start,matched+1,altered))
                continue
            quality = 1 - .075*gap - .20*altered
            if best is None or quality > best[0]:
                best = quality,start,i
        states = list(dict.fromkeys(active))
    if best is None:
        return None
    quality,start,end = best
    left = spans[max(0,start-4)].start()
    right = spans[min(len(spans)-1,end+14)].end()
    return quality, text[left:right].strip()


def rank_pass(con, original, forms, kinds, query):
    # Rank an OR retrieval and an all-topic retrieval. The latter keeps
    # documents covering all concepts eligible on a large, noisy index.
    all_terms = " OR ".join(dict.fromkeys(form.fts() for form in forms))
    core = [w for w in original if w not in GENRE] or original
    queries = [all_terms]
    if len(core) >= 2:
        by_group = ["("+" OR ".join(f.fts() for f in forms if g in f.groups)+")" for g in core]
        queries.append(" AND ".join(dict.fromkeys(by_group)))
    # A compact occurrence late in a long file can miss the top BM25
    # shortlist. FTS narrows the text window, Python verifies word order.
    near = None
    if 2 <= len(core) <= 8:
        needles = " ".join('"'+w+'"'+('*' if len(w)>=4 else '') for w in core)
        near = f"NEAR({needles}, 3)"
        queries.append(near)
    kind_sql = (" AND files.kind IN (" + ",".join("?" for _ in kinds) + ")") if kinds else ""
    pool = {}
    for match in queries:
        sql = """SELECT files.*, bm25(docs,8,1) AS relevance FROM docs
                 JOIN files ON files.id=docs.rowid WHERE docs MATCH ?""" + kind_sql + """
                 ORDER BY relevance,mtime DESC LIMIT 100"""
        for row in con.execute(sql,(match,*kinds)):
            item = pool.setdefault(row["id"],dict(row))
            if match == near:
                item["compact"] = True
    # Materialize text windows only for the shortlist; do not load the full
    # content of every textbook or OCR image into the Python process.
    anchor = {g for g in core if re.fullmatch("[a-z][a-z0-9]+",g)} if re.search("[а-яё]",query,re.I) else set()
    weights = {w: (0.35 if w not in core else 1.0) for w in original}
    total = sum(weights.values())

    def fraction(found):
        return sum(weights[g]*quality for g,quality in found.items())/total

    results = []
    for row in pool.values():
        window = con.execute("""SELECT substr(body,1,2300),snippet(docs,1,'','','…',32)
                                 FROM docs WHERE rowid=? AND docs MATCH ?""",(row["id"],all_terms)).fetchone()
        if window is None:
            continue
        path = Path(row["path"])
        snippet = window[1]
        if row.pop("compact", False):
            hit = con.execute("""SELECT snippet(docs,1,'','','…',32) FROM docs
                                 WHERE rowid=? AND docs MATCH ?""",(row["id"],near)).fetchone()
            if hit:
                snippet = hit[0]
        name = words(path.stem)
        in_name = evidence(forms,name)
        in_title = evidence(forms,name + words(header(row["kind"],window[0])))
        nearby = evidence(forms,words(snippet))
        # Prefer the repeated body's actual phrase to disconnected matches in
        # a title + snippet. Keep the literal OCR so the user can inspect it.
        image = row["kind"] in IMAGES
        phrase = phrase_in_text(core,snippet,image)
        if image and (phrase is None or phrase[0] < 1):
            heading = phrase_in_text(core,window[0][:650],True)
            if heading and (phrase is None or heading[0] > phrase[0]):
                phrase = heading
        if phrase:
            for concept in core:
                nearby.setdefault(concept,0.92)
        combined = {g:max(nearby.get(g,0),in_title.get(g,0)) for g in nearby.keys() | in_title.keys()}
        if not any(g in combined for g in core):
            continue
        # Remembered names, document headings, and a coherent passage
        # distinguish a file about a subject from a passing mention.
        score = (48*fraction(combined) + 24*fraction(in_title) + 16*fraction(in_name)
                 + 10*fraction(nearby) + min(3, -row["relevance"]*.1))
        if phrase:
            score += 38*phrase[0]
        if set(core).issubset(in_name):
            # Among complete filename matches prefer the more specific title
            # to a longer qualified variant when other evidence is similar.
            score += 5 / max(1,len(name))
        exact_name = path.name.casefold() == query.strip(" «»\"").casefold()
        if exact_name:
            score += 100
        full = set(core).issubset(combined)
        local = set(core).issubset(in_title) or set(core).issubset(nearby)
        genre_ok = (len(core)==len(original) or any(w in combined for w in original if w not in core)
                    or set(core).issubset(in_title))
        fuzzy = any(value < .9 for value in combined.values())
        row["weak"] = not exact_name and (not (full and local and genre_ok) or fuzzy)
        missing = anchor - combined.keys()
        row["missing_anchors"] = [w for w in WORD.findall(query) if w.casefold() in missing]
        if missing:
            score *= .55
        row["score"] = score
        row["excerpt"] = phrase[1] if phrase else (snippet if nearby else "")
        results.append(row)
    return results


def exact_files(con, query):
    name = re.sub(r"^(?:найди|найти|покажи|ищу|find)\s+(?:файл\s+)?","",query.strip(),flags=re.I).strip(' «»"')
    if not name:
        return []
    basename = Path(name).name
    terms = words(basename)
    if terms:
        # Bypass the ranked shortlist: even a filename made from frequent
        # words is conclusive. Verify punctuation and casing in Python below.
        term_query = 'name: "' + " ".join(WORD.findall(basename.lower())) + '"'
        candidates = con.execute("""SELECT files.* FROM docs JOIN files ON files.id=docs.rowid
                                    WHERE docs MATCH ?""",(term_query,))
    else:
        candidates = con.execute("SELECT * FROM files WHERE substr(path, -length(?)) = ?",("/"+basename,"/"+basename))
    return [dict(row,weak=False,score=200,excerpt="") for row in candidates
            if Path(row["path"]).name.casefold()==basename.casefold()
            and ("/" not in name or row["path"].endswith(name))]


def search(db, query, hints=(), type_hint=""):
    original = tokens(query)
    kinds = sorted(requested_types(query))
    with connect(db) as con:
        con.execute("BEGIN")  # both passes see the same index during updates
        exact = exact_files(con,query)
        if not original:
            if exact:
                return sorted(exact,key=lambda r:(-r["mtime"],r["path"]))[:10]
            if not kinds:
                return []
            slots = ",".join("?" for _ in kinds)
            return [dict(r,score=0,weak=False,excerpt="") for r in con.execute(
                f"SELECT * FROM files WHERE kind IN ({slots}) ORDER BY mtime DESC LIMIT 10",kinds)]
        lexical = [Form((w,), (w,)) for w in original]
        base = rank_pass(con, original, lexical, kinds, query)
        aliases = aliases_for(hints,original)
        extra = rank_pass(con,original,lexical+aliases,kinds,query) if aliases else []
        # Keep the best supporting evidence from either retrieval pass.
        ranked = {}
        for row in base + extra:
            best = ranked.get(row["id"])
            quality = (not row["weak"],row["score"])
            if best is None or quality > (not best["weak"],best["score"]):
                ranked[row["id"]] = row
        if not ranked and not exact:
            similar = []
            for w in original:
                if len(w) < 4:
                    continue
                vocabulary = [r[0] for r in con.execute("SELECT term FROM vocab WHERE term>=? AND term<? LIMIT 2000",(w[:2],w[:2]+chr(0x10ffff)))]
                similar += [Form((term,),(w,),fuzzy=True) for term in difflib.get_close_matches(w,vocabulary,n=2,cutoff=.78)]
            if similar:
                for row in rank_pass(con,original,lexical+similar,kinds,query):
                    ranked[row["id"]] = row

    def order(row):
        return (row["weak"], -row["score"], "backups" in Path(row["path"]).parts, -row["mtime"], row["path"])
    exact_ids = {r["id"] for r in exact}
    # Keep the first strong direct result stable. Let a translated title
    # compete with other documents that merely mention all the original words.
    direct = sorted((r for r in base if not r["weak"] and r["id"] not in exact_ids),key=order)
    protected = sorted(exact,key=order) + direct[:1]
    seen_ids = {r["id"] for r in protected}
    ordered = protected + sorted((r for i,r in ranked.items() if i not in seen_ids),key=order)
    # Give different filenames a chance before displaying backup copies.
    seen, primary, copies = set(), [], []
    for row in ordered:
        key = (Path(row["path"]).name.casefold(),row["size"])
        (copies if key in seen else primary).append(row)
        seen.add(key)
    return (primary+copies)[:10]
