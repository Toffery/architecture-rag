import re
from pathlib import Path
from statistics import mean

from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parent.parent
CHUNK_WORDS = 200
OVERLAP_WORDS = 25


def load_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_WORDS,
        chunk_overlap=OVERLAP_WORDS,
        length_function=lambda text: len(text.split()),
    )
    for path in sorted((ROOT / "Task2" / "knowledge_base").rglob("*.md")):
        document = path.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^#{1,6}[ \t]+([^\n]+)$", document, re.MULTILINE))
        title = path.stem
        if headings and headings[0].group().startswith("# "):
            title = headings[0].group(1).strip()

        sections = [("", 0)]
        sections.extend((heading.group(1).strip(), heading.end()) for heading in headings)
        ends = [heading.start() for heading in headings] + [len(document)]
        chunk_number = 0

        for (section, section_start), section_end in zip(sections, ends):
            section_text = document[section_start:section_end]
            offset = -1
            for text in splitter.split_text(section_text):
                offset = section_text.find(text, offset + 1)
                if offset < 0:
                    raise ValueError(f"Couldn't find chunk position in {path}")
                
                start = section_start + offset
                end = start + len(text)
                chunk_number += 1
                yield {
                    "text": document[start:end],
                    "title": title,
                    "section": section,
                    "path": path.relative_to(ROOT).as_posix(),
                    "chunk_number": chunk_number,
                    "start_char": start,
                    "end_char": end,
                }


if __name__ == "__main__":
    chunks = list(load_chunks())
    if not chunks:
        raise SystemExit("В Task2/knowledge_base/ нет текста для разбиения.")

    sizes = [len(chunk["text"].split()) for chunk in chunks]
    print(f"Документов: {len({chunk['path'] for chunk in chunks})}")
    print(f"Чанков: {len(chunks)}")
    print(f"Слов в чанке (мин. / среднее / макс.): {min(sizes)} / {mean(sizes):.1f} / {max(sizes)}")
    for chunk in chunks[:3]:
        print(f"\n{chunk['path']} — {chunk['section'] or chunk['title']}")
        print(f"Чанк {chunk['chunk_number']}, символы {chunk['start_char']}:{chunk['end_char']}")
        print(chunk["text"][:240])
