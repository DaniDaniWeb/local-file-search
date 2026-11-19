"""Offline passage search. File metadata and semantic passages share a SQLite snapshot."""
import fcntl
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import threading

import catalog

# Keep passages short enough for an encoder and the GUI excerpt. Split the
# entire stored text, with overlap for sentences that cross a boundary.
VERSION = "passage-v1"
BATCH = 24
_lock = threading.Lock()
_model = None
_STAMP = """s.file_id=f.id AND s.path=f.path AND s.mtime=f.mtime
            AND s.size=f.size AND s.ctime=f.ctime AND f.pending=0"""
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_files (
  file_id INTEGER PRIMARY KEY, path TEXT NOT NULL, mtime INTEGER NOT NULL,
  size INTEGER NOT NULL, ctime INTEGER NOT NULL, version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_cache (
  id INTEGER PRIMARY KEY, model TEXT NOT NULL, digest BLOB NOT NULL,
  passage TEXT NOT NULL, vector BLOB NOT NULL, UNIQUE(model,digest)
);
CREATE TABLE IF NOT EXISTS vector_links (
  file_id INTEGER NOT NULL, cache_id INTEGER NOT NULL,
  PRIMARY KEY(file_id,cache_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS vector_by_cache ON vector_links(cache_id);
"""


def _version():
    from encoder import MODEL_ID
    return MODEL_ID + "/" + VERSION


def _loaded_encoder():
    from encoder import Encoder
    global _model
    if _model is None:
        _model = Encoder()
    return _model


def _encode(texts, query=False):
    import numpy as np
    with _lock:
        values = np.asarray(_loaded_encoder().encode(texts, query=query), dtype=np.float32)
    if (values.ndim != 2 or values.shape[0] != len(texts)
            or values.shape[1] == 0 or not np.isfinite(values).all()):
        raise ValueError("Некорректные векторы модели поиска")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def normalized_title(name):
    return " ".join(re.findall(r"[^\W_]+", Path(name).stem))[:200]


def passages(name, body):
    title = normalized_title(name)
    if re.search(r"[^\W\d_]{2}", title):
        yield title
    # Normalize PDF line wraps/OCR whitespace before splitting and hashing.
    text = " ".join(body.split())
    if text:
        with _lock:
            model = _loaded_encoder()
        yield from model.split(text, max_tokens=160, overlap=16)


def _exists(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_files'").fetchone()


def status(db):
    """Number of current files and passage links usable by this encoder."""
    if not Path(db).is_file():
        return 0, 0
    with catalog.connect(db) as con:
        if not _exists(con):
            return 0, 0
        row = con.execute(f"""SELECT count(DISTINCT f.id),count(l.cache_id)
            FROM files f JOIN vector_files s ON {_STAMP}
            LEFT JOIN vector_links l ON l.file_id=f.id
            WHERE s.version=?""", (_version(),)).fetchone()
        return tuple(row)


def _flush(con, items, model, version, progress):
    """Reuse content-addressed vectors, including identical passages across files."""
    texts = {}
    links = []
    for row, body in items:
        digests = set()
        for passage in passages(row["path"], body):
            digest = hashlib.sha256(passage.encode()).digest()
            texts[digest] = passage
            digests.add(digest)
        links.append((row, digests))
    digests = list(texts)
    cache = {}
    for i in range(0, len(digests), 400):
        batch = digests[i:i + 400]
        slots = ",".join("?" for _ in batch)
        cache.update(con.execute(
            f"SELECT digest,id FROM vector_cache WHERE model=? AND digest IN ({slots})",
            [model, *batch]))
    # Group similarly sized passages to reduce padded CPU inference work.
    missing = sorted((digest for digest in digests if digest not in cache),
                     key=lambda d: len(texts[d]))
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i + BATCH]
        embeddings = _encode([texts[d] for d in batch])
        with con:
            for digest, embedding in zip(batch, embeddings):
                inserted = con.execute("""INSERT INTO vector_cache(model,digest,passage,vector)
                    VALUES(?,?,?,?)""", (model,digest,texts[digest],embedding.astype("<f2").tobytes()))
                cache[digest] = inserted.lastrowid
        if len(missing) > 4 * BATCH:
            progress(f"Фрагменты: {min(i+BATCH,len(missing))} из {len(missing)}")
    # Attach all passages for each file in one transaction. A simultaneous
    # text refresh can invalidate a file while the CPU is working.
    with con:
        con.execute("BEGIN IMMEDIATE")
        for row, doc_digests in links:
            stamp = tuple(row[k] for k in ("id","path","mtime","size","ctime"))
            current = con.execute("""SELECT 1 FROM files WHERE id=? AND path=?
                AND mtime=? AND size=? AND ctime=? AND pending=0""", stamp).fetchone()
            if current is None:
                continue
            con.execute("DELETE FROM vector_links WHERE file_id=?", (row["id"],))
            con.executemany("INSERT INTO vector_links(file_id,cache_id) VALUES(?,?)",
                            ((row["id"], cache[d]) for d in doc_digests))
            con.execute("""INSERT OR REPLACE INTO vector_files
                (file_id,path,mtime,size,ctime,version) VALUES(?,?,?,?,?,?)""", (*stamp,version))


def refresh(db, progress=lambda msg: None):
    """Called by the background index worker. Resumable after interruption."""
    from encoder import MODEL_ID
    db = Path(db)
    version = _version()
    with db.with_suffix(".semantic.lock").open("a") as lock, catalog.connect(db) as con:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The parallel caller may have just committed new OCR/text. Take
            # another snapshot after the running build so those changes stick.
            progress("Ожидание текущего обновления смыслового индекса")
            fcntl.flock(lock,fcntl.LOCK_EX)
        con.executescript(_SCHEMA)
        rows = con.execute(f"""SELECT f.* FROM files f LEFT JOIN vector_files s
            ON {_STAMP} AND s.version=? WHERE f.pending=0 AND s.file_id IS NULL
            ORDER BY CASE WHEN f.kind IN ('png','jpg','jpeg','md','markdown') THEN 0
                          WHEN f.kind='pdf' THEN 2 ELSE 1 END, f.size""", (version,)).fetchall()
        bundle = []
        volume = 0
        for i, row in enumerate(rows,1):
            # Name+body and stamp come from the same read snapshot, even
            # when FTS refresh or deletion interleaves with semantic indexing.
            content = con.execute("""SELECT body FROM docs JOIN files ON files.id=docs.rowid
                WHERE id=? AND path=? AND mtime=? AND size=? AND ctime=? AND pending=0""",
                tuple(row[k] for k in ("id","path","mtime","size","ctime"))).fetchone()
            if content is not None:
                bundle.append((row,content[0]))
                volume += len(content[0])
            if bundle and (len(bundle)>=24 or volume>=30_000 or i==len(rows)):
                progress(f"Смысловой индекс · файлы {i-len(bundle)+1}–{i} из {len(rows)}")
                _flush(con,bundle,MODEL_ID,version,progress)
                bundle.clear()
                volume = 0
        with con:
            con.execute(f"""DELETE FROM vector_files AS s WHERE NOT EXISTS
                (SELECT 1 FROM files f WHERE {_STAMP})""")
            con.execute("DELETE FROM vector_links WHERE file_id NOT IN (SELECT file_id FROM vector_files)")
            con.execute("DELETE FROM vector_cache WHERE id NOT IN (SELECT cache_id FROM vector_links)")
    files, count = status(db)
    return f"Смысловой индекс · файлов {files} · фрагментов {count}"


def search(db, query, limit=20, *, description=None):
    """Read-only cosine search. Return the best current passage for each file."""
    import numpy as np
    if limit <= 0 or not query.strip() or not Path(db).is_file():
        return []
    with catalog.connect(db) as con:
        if not _exists(con):
            return []
        # One consistent view for vectors, paths and snippets during refresh.
        con.execute("BEGIN")
        kinds = sorted(catalog.requested_types(query))
        description = (description or query).strip()
        description = re.sub(r"^(?:найди(?:те)?|покажи(?:те)?|ищу|find)\s+", "", description, flags=re.I)
        if not catalog.tokens(description):
            return []
        description = re.sub(r"^(?:pdf|markdown|скриншот|файл)\s+(?:(?:с|про|по)\s+)?","",description,flags=re.I)
        kind_sql = (" AND f.kind IN (" + ",".join("?" for _ in kinds) + ")") if kinds else ""
        meta = {r["id"]:dict(r) for r in con.execute(f"""SELECT f.* FROM files f
            JOIN vector_files s ON {_STAMP} WHERE s.version=? {kind_sql}""",(_version(),*kinds))}
        # Title vectors are already in the shared cache. A document with a
        # descriptive title can compete separately from long passage lists.
        title_keys, meaningful = {}, set()
        generic = {"screenshot","from","img","dsc","pasted","image","preview","photo","png","jpg","скриншот"}
        for ident,row in meta.items():
            title = normalized_title(row["path"])
            title_keys[ident] = hashlib.sha256(title.encode()).digest()
            tokens = catalog.words(title)
            opaque = any(len(w) >= 20 and re.search("[0-9]",w) for w in tokens)
            if not opaque and any(2 <= len(w) <= 28 and w.isalpha() and w not in generic for w in tokens):
                meaningful.add(ident)
        cursor = con.execute(f"""SELECT f.id,v.id,v.vector,v.digest
            FROM vector_files s JOIN files f ON {_STAMP}
            JOIN vector_links l ON l.file_id=s.file_id JOIN vector_cache v ON v.id=l.cache_id
            WHERE s.version=? {kind_sql}""", (_version(),*kinds))
        batch = cursor.fetchmany(384)
        if not batch:
            return []
        query_vector = _encode([description],query=True)[0]
        best, titles = {}, {}
        while batch:
            raw = b"".join(r[2] for r in batch)
            matrix = np.frombuffer(raw,dtype="<f2").reshape(len(batch), len(query_vector))
            scores = np.einsum("ij,j->i",matrix,query_vector,dtype=np.float32)
            for row, score in zip(batch,scores):
                ident, cache_id = row[0],row[1]
                is_title = row[3] == title_keys[ident]
                if is_title and ident not in meaningful:
                    continue
                if ident not in best or score>best[ident][0]:
                    best[ident] = (float(score),cache_id)
                if is_title:
                    titles[ident] = (float(score),cache_id)
            batch = cursor.fetchmany(384)
        def rank_key(item):
            ident,(similarity,_) = item
            path = Path(meta[ident]["path"])
            return (-similarity,"backups" in path.parts,ident)
        rank_scores, ranks = {}, {}
        for channel,candidates,weight in (("passage_rank",best,1.0),("title_rank",titles,.85)):
            for rank,(ident,_) in enumerate(sorted(candidates.items(),key=rank_key),1):
                rank_scores[ident] = rank_scores.get(ident,0) + weight / (4+rank)
                ranks.setdefault(ident,{})[channel] = rank
        seen, unique, copies = set(), [], []
        for ident in sorted(rank_scores,key=rank_scores.get,reverse=True):
            row = meta[ident]
            key = (Path(row["path"]).name.casefold(),row["size"])
            (copies if key in seen else unique).append(ident)
            seen.add(key)
        results = []
        for ident in (unique+copies)[:limit]:
            score,cache_id = best[ident]
            excerpt = con.execute("SELECT passage FROM vector_cache WHERE id=?",(cache_id,)).fetchone()[0]
            results.append(dict(meta[ident],**ranks[ident],excerpt=excerpt,score=score,similarity=score,weak=True,semantic=True))
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Обновить локальный смысловой индекс")
    parser.add_argument("--db", type=Path, default=catalog.DB)
    args = parser.parse_args()
    os.umask(0o077)
    print(refresh(args.db,lambda s: print(s,flush=True)),flush=True)
