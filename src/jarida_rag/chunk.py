import bisect
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .pdf import file_hash, read_manifest, read_pages

TITLE = re.compile(
    r"^(?:ظهير شريف|مرسوم|قرار|مقرر|قانون)[^.]{0,200}?رقم\s*(\d[\d.]*\d)\s+صادر",
    re.M,
)
TEXT_END = re.compile(r"حرر ب")
# A real new text starts right after the signature block of the previous one.
# A decree quoted in the middle of a long text does not, so it is not a new text.
MAX_SIGNATURE_GAP = 1500


def find_titles(full_text):
    ends = [match.end() for match in TEXT_END.finditer(full_text)]
    titles = []
    for title in TITLE.finditer(full_text):
        previous = bisect.bisect_left(ends, title.start())
        last_end = ends[previous - 1] if previous else None
        right_after_an_end = (
            last_end is not None
            and (not titles or last_end > titles[-1].start())
            and title.start() - last_end < MAX_SIGNATURE_GAP
        )
        if not titles or right_after_an_end:
            titles.append(title)
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
    bounds = [title.start() for title in titles] + [len(full_text)]

    chunks = []
    for title_match, start, end in zip(titles, bounds, bounds[1:]):
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
                "text_number": title_match.group(1),
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
