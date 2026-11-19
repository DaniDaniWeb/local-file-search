"""Optional query expansion via a loopback-only chat completions endpoint."""
import http.client
import json
import re
from urllib.parse import urlsplit


def settings(value):
    if not isinstance(value, dict):
        raise ValueError("The qwen setting must be null or an object with base_url and model")
    base, model = value.get("base_url"), value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Set qwen.model to the name of the model served locally")
    if not isinstance(base, str):
        raise ValueError("Set qwen.base_url to a local HTTP URL")
    url = urlsplit(base)
    host = url.hostname
    if (url.scheme != "http" or host not in ("localhost", "127.0.0.1", "::1")
            or url.username is not None or url.query or url.fragment or url.port == 0):
        raise ValueError("Qwen requires a loopback HTTP URL without credentials, a query, or a fragment")
    return ("127.0.0.1" if host == "localhost" else host), url.port or 80, url.path.rstrip("/"), model.strip()


# Words about the act of searching and the medium belong to the query parser.
# Translating them into content terms gives generic documents a false advantage.
NON_SUBJECT = set("""
    найди найти покажи отыщи поищи хочу нужна нужен нужно мой моя мне файл файлы
    где про по на с со в и о об от до из для между после перед при к за или а во pdf png jpeg jpg md markdown документ
    скачать пожалуйста ищу страницы скан пдф фото снимок
    find search show look file files image images screenshot screenshots photo photos
    note notes textbook picture pictures document about on with of the for a an in
    containing contains please can could you i me my named called where which is
""".split())
MEDIUM = ("скриншот", "конспект", "учебник", "картинк", "фотограф", "документ", "заметк")
WORD = re.compile(r"[^\W_]+")
PHRASE = re.compile(r"[\w+#-]+(?: [\w+#-]+){0,5}")


# Separate format/command words before calling the model. Numbered concepts give
# short inputs as much context as long sentences and keep Russian inflections intact.
def subjects(query):
    groups = []
    current = []
    end = 0
    for match in WORD.finditer(query):
        word = match.group()
        folded = word.casefold()
        separator = query[end:match.start()].strip()
        end = match.end()
        ignored = folded in NON_SUBJECT or folded.startswith(MEDIUM)
        switches_script = bool(current) and current[-1].isascii() != word.isascii()
        if ignored or separator or switches_script or len(current) == 4:
            if current:
                groups.append(current)
            current = []
        if not ignored:
            current.append(word)
    if current:
        groups.append(current)
    # Retain the whole phrase (e.g. "machine learning") and component terms
    # for the same concept in an inflected or longer title.
    phrases = [" ".join(group) for group in groups]
    words = [word for group in groups if len(group) > 1 for word in group]
    return list(dict.fromkeys(p for p in phrases + words if len(p) <= 64))[:10]


def locate_source(source, query):
    literal = re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", query, re.I)
    if literal:
        return literal.group()
    # Some responses use nominative keys. Recover the actual contiguous source
    # span only when the Russian words differ by common case endings.
    def inflection(word):
        word = word.casefold().replace("ё", "е")
        if re.fullmatch("[а-я]{4,}", word):
            return re.sub(r"(иями|ыми|ими|ами|ями|его|ого|ому|ему|ую|юю|ой|ая|яя|ое|ее|ый|ий|ые|ие|ом|ем|ов|ах|ых|ям|ия|ии|ию|[аеуяыию])$", "", word)
        return word
    source_words = [inflection(w) for w in WORD.findall(source)]
    words = list(WORD.finditer(query))
    for i in range(len(words) - len(source_words) + 1):
        group = words[i:i+len(source_words)]
        if group and [inflection(m.group()) for m in group] == source_words:
            return query[group[0].start():group[-1].end()]
    return None


def aliases_from_reply(reply, query, ids=None):
    """Accept only short, source-grounded mappings; a model cannot supply filters or SQL."""
    clean = reply.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.I)
    obj = json.loads(clean)
    if not isinstance(obj, dict):
        raise ValueError("Qwen returned an invalid translation map")

    aliases = {}
    for source, translations in obj.items():
        source = (ids or {}).get(source, source)
        if isinstance(translations, str):
            translations = [translations]
        if not isinstance(translations, list) or len(source) > 64 or not PHRASE.fullmatch(source):
            continue
        actual_source = locate_source(source, query)
        if not actual_source:
            continue
        source_words = WORD.findall(source.casefold())
        if (not source_words or source_words[0] in NON_SUBJECT
                or any(word.startswith(MEDIUM) for word in source_words)):
            continue
        options = []
        for value in translations:
            if not isinstance(value, str):
                continue
            value = value.strip()
            if (len(value) > 60 or not PHRASE.fullmatch(value)
                    or value.casefold() == source.casefold() or value.casefold() in NON_SUBJECT):
                continue
            if value.casefold() not in {option.casefold() for option in options}:
                options.append(value)
            if len(options) == 2:
                break
        if options:
            aliases[actual_source] = options
        if len(aliases) == 10:
            break
    return aliases


def expand(query, options, status):
    ids = {str(i): phrase for i, phrase in enumerate(subjects(query), start=1)}
    if not ids:
        return {}, ""
    host, port, path, model = settings(options)
    status("Qwen is translating search terms · " + model)
    prompt = (
        "Translate each numbered expression from English to Russian or from Russian to English. "
        "Use the full query to interpret the meaning and any technical jargon. Expand Latin-letter "
        "abbreviations into their established English names and preserve proper names. "
        "Return only a JSON object with the same numbered keys and one short translation per key. "
        'For example, {"1":"supply chain","2":"CPU"} becomes '
        '{"1":"цепочка поставок","2":"central processing unit"}. '
        "Full query for context. " + json.dumps(query[:2000], ensure_ascii=False) + ". "
        "Expressions to translate. "
        + json.dumps(ids, ensure_ascii=False)
    )
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "temperature": 0.1, "max_tokens": 512,
    }).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=12)
    try:
        conn.request("POST", path + "/chat/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"The model server returned HTTP {response.status}")
        result = json.loads(response.read(262144))
        message = result["choices"][0]["message"]["content"]
        if not isinstance(message, str):
            raise ValueError("The model did not return translation text")
        return aliases_from_reply(message, query, ids), ""
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The model returned an invalid response") from exc
    finally:
        conn.close()
