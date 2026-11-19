"""Rank fusion for lexical results and passages retrieved in either language."""
from pathlib import Path
import re

import catalog


def translated_description(query, hints):
    """Read source spans left to right; the longest matching translation wins."""
    if not hints:
        return ""
    options = {word.casefold(): equivalents[0] for word,equivalents in hints.items() if word and equivalents}
    if not options:
        return ""
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(options, key=len, reverse=True)) + r")(?!\w)"
    phrases = [options[m.group().casefold()] for m in re.finditer(pattern,query,re.I)]
    return " ".join(dict.fromkeys(phrases))


def _names(query):
    terms = catalog.WORD.findall(query)
    ignored = catalog.FILLER | catalog.FORMAT
    return [word for pos, word in enumerate(terms)
            if word.lower() not in ignored and len(word) > 1
            and (word.isupper() or (pos > 0 and word[0].isupper()))]


def combine(query, lexical, semantic, translated=()):
    if not semantic and not translated:
        return lexical
    kinds = catalog.requested_types(query)
    names = _names(query)
    by_path = {row["path"]: row for row in lexical}
    scores = {row["path"]: (0.85 if row["weak"] else 1.4) / (4 + rank)
              for rank, row in enumerate(lexical, 1)}
    best_passage = {}
    translated_title = None
    for rows, weight in ((semantic,1.0),(translated,.8)):
        for rank, row in enumerate(rows, 1):
            if kinds and row["kind"] not in kinds:
                continue
            path = row["path"]
            evidence = catalog.words(Path(path).stem + " " + row["excerpt"])
            missing = [name for name in names if not any(w.startswith(catalog.stem(name.casefold())) for w in evidence)]
            initials = "".join(n[0].casefold() for n in names)
            if missing and not (len(initials) > 1 and initials in evidence):
                continue
            strength = weight / (4 + rank)
            scores[path] = scores.get(path, 0) + strength
            if (rows is translated and rank == 1 and row.get("title_rank",100) <= 3
                    and row.get("passage_rank",100) <= 12):
                translated_title = path
            previous = by_path.get(path)
            if (previous is None or previous["weak"]) and strength > best_passage.get(path,0):
                by_path[path] = dict(row, semantic=True, weak=True)
                best_passage[path] = strength

    order = sorted(scores,key=scores.get,reverse=True)
    pinned = [r["path"] for r in lexical if r["score"] >= 200]
    if not pinned and lexical and not lexical[0]["weak"]:
        pinned = [lexical[0]["path"]]
    ordered = pinned + [p for p in order if p not in pinned]
    if lexical and lexical[0]["path"] not in ordered[:2]:
        # Keep the leading wording-based result directly accessible even
        # when a paraphrase has several high-ranked semantic neighbours.
        path = lexical[0]["path"]
        ordered.remove(path)
        ordered.insert(1,path)
    if translated_title and not pinned and translated_title not in ordered[:3]:
        ordered.remove(translated_title)
        ordered.insert(2,translated_title)
    seen, unique, copies = set(), [], []
    for path in ordered:
        row = by_path[path]
        key = (Path(path).name.casefold(),row["size"])
        (copies if key in seen else unique).append(row)
        seen.add(key)
    results = (unique + copies)[:10]
    # Leave room for earlier text matches when many weak semantic neighbours
    # arrive. Identical backup filenames need only one slot in this safeguard.
    keep = {(Path(r["path"]).name.casefold(),r["size"]) for r in lexical[:3]}
    for row in lexical[:3]:
        keys = [(Path(r["path"]).name.casefold(),r["size"]) for r in results]
        if (Path(row["path"]).name.casefold(),row["size"]) not in keys:
            slot = next(i for i in reversed(range(len(keys))) if keys[i] not in keep)
            results[slot] = row
    return results
