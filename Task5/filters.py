"""Фильтрация найденных фрагментов перед сборкой промпта."""

import json
import re
from datetime import datetime
from pathlib import Path


COMMAND = re.compile(
    r"\bignore\s+(?:(?:all|previous|prior|system)\s+)*instructions\b[.!]?"
    r"|\b(?:игнорируй(?:те)?|забудь(?:те)?)\s+"
    r"(?:(?:все|предыдущие|прошлые|системные)\s+)*(?:инструкции|указания)\b[.!]?"
    r"|(?<!\w)/system\b",
    re.IGNORECASE,
)
SECRET = re.compile(
    r"\b(?:password|passwd|(?:супер)?пароль|api[_ -]?key)"
    r"(?:\s+(?:root|администратора))?\s*[:=]\s*[^\s\n]",
    re.IGNORECASE,
)


LOG_PATH = Path(__file__).with_name("filters.log")


def strip_commands(chunks, events):
    cleaned = []
    for chunk in chunks:
        lines = chunk["text"].splitlines(keepends=True)
        kept_lines = [line for line in lines if not COMMAND.search(line)]
        count = len(lines) - len(kept_lines)
        if count:
            text = "".join(kept_lines).strip()
            events.append({"path": chunk["path"], "action": "команды удалены", "count": count})
            if not text:
                continue
            chunk = {**chunk, "text": text}
        cleaned.append(chunk)
    return cleaned


def filter_chunks(chunks, mode="drop", *, question=None, strip=False):
    events = []
    found = len(chunks)
    if strip:
        chunks = strip_commands(chunks, events)
    if mode == "off":
        kept, flagged = chunks, []
        events.append({"action": "проверка отключена"})
    else:
        kept, flagged = check_chunks(chunks, events)
    if question is not None:
        entry = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "question": "[скрыт: похож на запись секрета]" if SECRET.search(question) else question,
            "mode": mode,
            "found": found,
            "checks": events,
            "kept": len(kept),
            "flagged": len(flagged),
        }
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return kept, flagged


def check_chunks(chunks, events):
    kept = []
    flagged = []
    for chunk in chunks:
        if SECRET.search(chunk["text"]):
            flagged.append((chunk["path"], "строка, похожая на раскрытие секрета"))
            events.append({"path": chunk["path"], "action": "отброшен", "match": "[запись секрета скрыта]"})
            continue
        command = COMMAND.search(chunk["text"])
        if not command:
            kept.append(chunk)
            events.append({"path": chunk["path"], "action": "пропущен в контекст"})
            continue
        flagged.append((chunk["path"], "команда внутри документа"))
        events.append({"path": chunk["path"], "action": "отброшен", "match": command.group()[:120]})
    return kept, flagged
