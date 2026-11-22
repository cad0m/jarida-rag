import bisect
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .pdf import file_hash, read_manifest, read_pages

LEGAL_TITLE = re.compile(
    r"^(?:ظهير شريف|مرسوم|قرار|مقرر|قانون)[^.]{0,200}?رقم\s*(\d[\d.]*\d)\s+صادر",
    re.M,
)
TEXT_END = re.compile(r"حرر\s*ب")

# The last section of an issue holds studies, notices and extracts: no decree number, no signature.
# "إعلانات وبلاغات" is often extracted as "إعالنات وبالغات".
ANNOUNCEMENTS_SECTION = re.compile(r"إع(?:ال|لا)نات و(?:بال|بلا)غات")
ANNOUNCEMENT_TITLE = re.compile(r"^(?:دراسة|مستخرج من)", re.M)


def find_titles(full_text):
    """Return (start offset, text number) for every legal text; announcements have no number."""
    section = ANNOUNCEMENTS_SECTION.search(full_text)
    legal_part_end = section.start() if section else len(full_text)

    titles = []
    for match in LEGAL_TITLE.finditer(full_text, 0, legal_part_end):
        # a decree quoted inside a text is not a new text: the previous one must be signed first
        if not titles or TEXT_END.search(full_text, titles[-1][0], match.start()):
            titles.append((match.start(), match.group(1)))

    if section:
        titles += [(match.start(), "") for match in ANNOUNCEMENT_TITLE.finditer(full_text, section.end())]
    return titles


def chunk_document(pages, meta):
    full_text = ""
    page_starts = []
    for page in pages:
        page_starts.append(len(full_text))
        full_text += page["text"] + "\n"
    page_numbers = [page["page"] for page in pages]

    def page_at(offset):
        return page_numbers[bisect.bisect_right(page_starts, offset) - 1]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\nالمادة ", "\n\n", "\n", ". ", "،", " ", ""],
        add_start_index=True,
    )

    titles = find_titles(full_text)
    ends = [start for start, _ in titles[1:]] + [len(full_text)]

    chunks = []
    for (start, text_number), end in zip(titles, ends):
        legal_text = full_text[start:end].rstrip()
        title = " ".join(legal_text[:200].split())

        for i, piece in enumerate(splitter.create_documents([legal_text])):
            text = piece.page_content
            if len(text) < config.MIN_CHUNK_CHARS:
                continue
            offset = start + piece.metadata["start_index"]
            chunks.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['doc_id']}:{len(chunks)}")),
                "text": text if i == 0 else f"{title}\n{text}",
                **meta,
                "title": title,
                "text_number": text_number,
                "page_start": page_at(offset),
                "page_end": page_at(offset + len(text) - 1),
            })
    return chunks


def chunk_all():
    chunks = []
    for row in read_manifest():
        path = config.PDF_DIR / row["filename"]
        meta = {"doc_id": file_hash(path), "issue": row["issue"], "date": row["date"], "lang": row["lang"]}
        chunks += chunk_document(read_pages(path, row["issue"]), meta)
    return chunks


if __name__ == "__main__":
    chunks = chunk_all()
    texts = {}
    for chunk in chunks:
        texts.setdefault((chunk["issue"], chunk["title"]), []).append(chunk)

    print(f"{len(chunks)} chunks, {len(texts)} legal texts\n")
    for (issue, title), parts in texts.items():
        pages = f"p.{parts[0]['page_start']}-{parts[-1]['page_end']}"
        print(f"BO {issue} {pages:10} {len(parts):3} chunks | {title[:90]}")
