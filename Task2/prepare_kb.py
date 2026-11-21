"""Скачать одну статью или весь список, очистить текст и заменить названия."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer

PAGE = None
# Только эти русские существительные склоняем автоматически.
INFLECT_TERMS = {"дух", "крип", "суперкрип", "мегакрип", "руна", "Мясник", "Могул", "Хан"}
PAUSE = 3
TIMEOUT = 30
ROOT = Path(__file__).resolve().parent.parent
API = "https://dota2.fandom.com/ru/api.php"

HEROES = [
    "Invoker", "Pudge", "Meepo", "Phantom Assassin", "Axe", "Sniper", "Mars",
    "Slardar", "Anti-Mage", "Troll Warlord", "Crystal Maiden", "Void Spirit",
    "Dazzle", "Magnus", "Storm Spirit", "Earth Spirit", "Vengeful Spirit",
    "Queen of Pain", "Morphling", "Wraith King",
]
ITEMS = [
    "Blink Dagger", "Black King Bar", "Aghanim's Scepter", "Force Staff",
    "Eul's Scepter of Divinity", "Town Portal Scroll", "Observer Ward", "Sentry Ward",
]
MECHANICS = [
    "Золото", "Опыт", "Броня", "Типы урона", "Обзор", "Развеивание",
    "Сопротивление магии", "Руны",
]
PAGES = HEROES + ITEMS + ["Постройки", "Фонтан", "Аванпост"] + MECHANICS
BUILDINGS = ["Башни", "Древние", "Казармы"]


def request(params):
    url = API + "?" + urlencode({"format": "json", **params})
    for attempt in range(2):
        time.sleep(PAUSE)
        try:
            req = Request(url, headers={"User-Agent": "ArchitectureRAG-study/0.1"})
            with urlopen(req, timeout=TIMEOUT) as response:
                data = json.load(response)
            if "error" in data:
                raise RuntimeError(f"Ошибка API: {data['error']}. Загрузка остановлена.")
            return data
        except HTTPError as error:
            if error.code in (403, 429):
                retry = error.headers.get("Retry-After", "не указан")
                raise RuntimeError(f"HTTP {error.code}. Стоп. Retry-After: {retry}") from error
            if error.code < 500 or attempt == 1:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 1:
                raise
        print("Временная ошибка сети. Одна повторная попытка через 30 секунд.")
        time.sleep(30)


def plain(node):
    text = node.get_text(" ", strip=True)
    text = re.sub(r"\[\s*\d*\s*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def clean(html, title, section=None):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output")
    if root is None:
        raise ValueError(f"Не найден текст статьи: {title}")
    for tag in root.select(
        "script, style, audio, video, img, svg, kbd, .mw-editsection, .toc, "
        ".navbox, .page-tabber, .infobox, .reference, .references, .gallery, "
        ".ext-audiobutton, .noprint"
    ):
        tag.decompose()

    if title in HEROES:
        parts = []
        bio = root.select_one("#heroBio")
        if bio:
            for row in bio.select('div[style*="display: table-row"]'):
                if plain(row).startswith("Голос:"):
                    row.decompose()
            parts += ["## Биография", plain(bio)]
        for ability in root.select(".ability-background"):
            description = ability.select_one(".ability-description")
            if not description:
                continue
            name = next(ability.stripped_strings, "Способность")
            parts.append("## " + name)
            blocks = description.find_all("div", recursive=False)
            parts.extend(plain(block) for block in blocks if plain(block))
            if not blocks:
                parts.append(plain(description))
        if not parts:
            raise ValueError(f"Не найдены биография и способности: {title}")
        return "\n\n".join(parts)

    parts = []
    active = section is None
    found = section is None
    skip = False
    for node in root.find_all(["h2", "h3", "h4", "p", "li", "tr"]):
        if node.find_parent(["p", "li", "tr"]):
            continue
        text = plain(node)
        if not text:
            continue
        if node.name == "h2":
            if section:
                active = text == section
                found = found or active
            skip = text in ["Недавние изменения", "История изменений", "Галерея",
                            "Ссылки", "См. также", "Примечания и ссылки"]
        if not active or skip:
            continue
        if node.name.startswith("h"):
            text = "## " + text
        elif node.name == "tr":
            text = " | ".join(plain(c) for c in node.find_all(["th", "td"], recursive=False))
        if text and (not parts or parts[-1] != text):
            parts.append(text)
    if not found or not parts:
        raise ValueError(f"Не найден непустой раздел: {title} / {section}")
    return "\n\n".join(parts)



def replace_terms(text, terms):
    replacements = {key.casefold(): value for key, value in terms.items()
                    if key.casefold() != value.casefold()}
    generated = {}
    morph = MorphAnalyzer()
    for source in INFLECT_TERMS:
        target = terms[source]
        if not re.fullmatch(r"[а-яё]+", target, re.IGNORECASE):
            raise ValueError(f"Для склонения нужно одно русское слово: {target!r}")
        targets = [p for p in morph.parse(target) if p.tag.POS == "NOUN"]
        if not targets:
            raise ValueError(f"Не удалось разобрать существительное: {target}")
        target_parse = targets[0]
        sources = [p for p in morph.parse(source)
                   if p.tag.POS == "NOUN" and p.normal_form == source.casefold()
                   and p.tag.animacy == target_parse.tag.animacy]
        if not sources:
            raise ValueError(f"Не удалось подобрать склонение для {source!r}")
        for form in sources[0].lexeme:
            inflected = target_parse.inflect({form.tag.case, form.tag.number})
            if inflected is None:
                raise ValueError(f"Не удалось склонить {target!r} как {form.word!r}")
            generated[form.word] = inflected.word
    combined = {**generated, **replacements}
    if not combined:
        return text
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(key) for key in sorted(combined, key=len, reverse=True)
    ) + r")(?!\w)", re.IGNORECASE)

    def substitute(match):
        word = match.group()
        result = combined[word.casefold()]
        if word.casefold() in generated:
            if word.isupper():
                return result.upper()
            if word.istitle():
                return result.capitalize()
            return result.lower()
        return result

    return pattern.sub(substitute, text)


def main():
    if PAGE is not None and PAGE not in PAGES:
        raise ValueError(f"Неизвестная страница {PAGE!r}. Выберите название из PAGES.")
    raw_dir = ROOT / "data/raw"
    clean_dir = ROOT / "data/clean"
    output_dir = ROOT / "Task2" / "knowledge_base"
    for folder in [raw_dir, clean_dir, output_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    terms = json.loads((ROOT / "Task2" / "terms_map.json").read_text(encoding="utf-8"))

    site_file = ROOT / "data/siteinfo.json"
    if not site_file.exists():
        site = request({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})
        site_file.write_text(json.dumps(site, ensure_ascii=False, indent=2), encoding="utf-8")

    report_file = ROOT / "data/review.json"
    review = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    for title in ([PAGE] if PAGE else PAGES):
        cache = raw_dir / (title + ".json")
        if cache.exists():
            print(f"Из кэша: {title}")
            saved = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"Скачиваем: {title}")
            data = request({"action": "parse", "page": title, "redirects": 1,
                            "prop": "text|revid|sections|links"})
            saved = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "data": data}
            cache.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        page = saved["data"]["parse"]
        source = "https://dota2.fandom.com/ru/wiki/" + quote(page["title"].replace(" ", "_"))
        for section in (BUILDINGS if title == "Постройки" else [None]):
            name = section or title
            text = "# " + name + "\n\n" + clean(page["text"]["*"], title, section) + "\n"
            (clean_dir / (name + ".md")).write_text(text, encoding="utf-8")
            if name not in terms:
                raise ValueError(f"Добавьте {name!r} в terms_map.json")
            result = replace_terms(text, terms)
            filename = terms[name] + ".md"
            (output_dir / filename).write_text(result, encoding="utf-8")
            candidates = sorted({link["*"] for link in page.get("links", [])
                                 if link.get("ns") == 0 and len(link["*"]) > 2
                                 and link["*"].casefold() in result.casefold()})
            review[name] = {"file": filename, "source": source,
                            "section": section, "page_id": page["pageid"],
                            "revision_id": page["revid"],
                            "downloaded_at": saved["downloaded_at"],
                            "authors": source + "?action=history",
                            "license": "См. data/siteinfo.json",
                            "terms_to_review": candidates}
            report_file.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Сохранён: knowledge_base/{filename} ({len(result)} символов)")
    print("Готово. Проверьте тексты и data/review.json; словарь замен дополняется вручную.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Остановлено: {error}")
