import bisect
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .pdf import file_hash, read_manifest, read_pages

LEGAL_TITLE = re.compile(
    r"^(?:بموجب )?(ظهير شريف|مرسوم|قرار|مقرر|قانون)[^.]{0,200}?رقم\s*(\d[\d.]*\d)\s+صادر",
    re.M,
)
# A text ends with its signature, a "* * *" separator between short notices, a new section header,
# or an erratum ("استدراك") correcting an earlier issue.
TEXT_END = re.compile(r"حرر\s*ب|\*\s*\*|^نصوص (?:عامة|خاصة)|^استدراك", re.M)

# The last section of an issue holds studies, notices and extracts: no decree number, no signature.
# "إعلانات وبلاغات" is often extracted as "إعالنات وبالغات".
ANNOUNCEMENTS_SECTION = re.compile(r"إع(?:ال|لا)نات و(?:بال|بلا)غات")
ANNOUNCEMENT_TITLE = re.compile(r"^(?:دراسة|مستخرج من)", re.M)
# "رأي" (opinion) also starts lines that quote other opinions, so it only counts right after the section header
FIRST_ANNOUNCEMENT = re.compile(r"^(?:دراسة|مستخرج من|رأي)", re.M)

# An article heading is alone on its line ("المادة 2") or followed by a dash ("المادة الثانية. - ...").
# Quoted articles of an amended text start with "»" and references like "المادة 16 من القانون" go on, so both are ignored.
ARTICLE_HEADING = re.compile(r"^المادة[ \t]+([^\s.]+)\.?[ \t]*(?:-|$)", re.M)

# The heading of a legal text stops where the issuer ("رئيس الحكومة،", "وزير ...،") or the preamble starts
HEADING_END = re.compile(r"\n(?:رئيس الحكومة|وزير|الوزير|وزيرة|الوزيرة|المندوب|بناء على|بعد الاط|إن )")
# the text's own date ("صادر في ... (...)"), not the dates of the texts it cites ("الصادر في ...")
DATE_CLAUSE = re.compile(r"(?<!ال)صادر\s*في[^)]*\)")
# The preamble (laws the text is based on) ends with the decision formula, before the first article
DECISION = re.compile(r"(?:رسم|قرر|قررت|قررا|يقرر) ما يلي\s*:?")
# A PDF line break that doesn't end a sentence and isn't followed by an article or a list item is just line wrap
# texts a title says it amends, completes or repeals ("بتغيير وتتميم المرسوم رقم 2.11.89")
AMENDS = re.compile(r"(?:بتغيير|بتتميم|بنسخ)[^\d]{0,60}?رقم\s*(\d[\d.]*\d)")
LINE_WRAP = re.compile(r"(?<![.:؛])\n(?!المادة|[-•*]|\d+\s*[-.)])")


def find_titles(full_text):
    """Return (start offset, text number, text type) for every legal text; announcements have no number."""
    section = ANNOUNCEMENTS_SECTION.search(full_text)
    legal_part_end = section.start() if section else len(full_text)

    titles = []
    for match in LEGAL_TITLE.finditer(full_text, 0, legal_part_end):
        # a decree quoted inside a text is not a new text: the previous one must be signed first
        if not titles or TEXT_END.search(full_text, titles[-1][0], match.start()):
            titles.append((match.start(), match.group(2), match.group(1)))

    first = FIRST_ANNOUNCEMENT.search(full_text, section.end()) if section else None
    if first:
        titles.append((first.start(), "", "إعلان"))
        titles += [(match.start(), "", "إعلان") for match in ANNOUNCEMENT_TITLE.finditer(full_text, first.end())]
    return titles


def heading(legal_text, text_number):
    if config.CHUNK_HEADER == "first_200":  # the old behaviour, kept for comparison
        return " ".join(legal_text[:200].split())
    if not text_number:  # announcements: their first line
        return legal_text.split("\n", 1)[0][:160]
    # joint orders list several ministers inside the title: search for its end after the number, and allow a long title
    end = HEADING_END.search(legal_text, legal_text.find(text_number), 500)
    title = legal_text[:end.start() if end else 300]
    return " ".join(DATE_CLAUSE.sub("", title, count=1).split())[:250]


def sections(legal_text):
    """(start, end) of the parts chunked separately: the preamble, then the body."""
    decision = DECISION.search(legal_text) if config.SPLIT_PREAMBLE else None
    if not decision:
        return [(0, len(legal_text))]
    return [(0, decision.end()), (decision.end(), len(legal_text))]


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
        separators=["\nالمادة ", "\n", ". ", "؛ ", "، ", " ", ""],
        add_start_index=True,
    )

    titles = find_titles(full_text)
    ends = [start for start, _, _ in titles[1:]] + [len(full_text)]

    chunks = []
    for (start, text_number, text_type), end in zip(titles, ends):
        legal_text = full_text[start:end].rstrip()
        title = heading(legal_text, text_number)
        # shortcut: title only, so "تنسخ مقتضيات المادة 5" inside an article is missed; parse articles for amendment links
        amends = "|".join(number for number in AMENDS.findall(title) if number != text_number)
        headings = [(match.start(), match.group(1)) for match in ARTICLE_HEADING.finditer(legal_text)]
        heading_starts = [position for position, _ in headings]
        if config.UNWRAP_LINES:
            # same length (one character for one), so every offset and page number stays exact
            legal_text = LINE_WRAP.sub(" ", legal_text)

        for section_start, section_end in sections(legal_text):
            pieces = splitter.create_documents([legal_text[section_start:section_end]])
            for i, piece in enumerate(pieces):
                source_text = piece.page_content  # exactly as in the PDF; this is what the LLM reads
                if len(source_text) < config.MIN_CHUNK_CHARS:
                    continue
                position = section_start + piece.metadata["start_index"]
                offset = start + position
                # the article the chunk starts in; later articles inside the chunk keep their heading in the text
                previous = bisect.bisect_right(heading_starts, position)
                article = headings[previous - 1][1] if previous else ""
                chunks.append({
                    # issue + position: stable across runs of the same pipeline; any change to extraction or chunking
                    # settings moves offsets, and the table is rebuilt from scratch anyway
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['doc_id']}:{offset}")),
                    "text": source_text,
                    "embed_text": embed_text(title, article, source_text, first=(i == 0 and section_start == 0)),
                    **meta,
                    "title": title,
                    "text_number": text_number,
                    "text_type": text_type,
                    "amends": amends,
                    "article": article,
                    "page_start": page_at(offset),
                    "page_end": page_at(offset + len(source_text) - 1),
                })
    return chunks


def embed_text(title, article, source_text, first):
    """What gets embedded: a one-line label for context, then the text itself."""
    if config.CHUNK_HEADER == "first_200":  # old behaviour: the 200-character title on every chunk but the first
        return source_text if first else f"{title}\n{source_text}"
    label = f"{title} | المادة {article}" if article else title
    return f"{label}\n{source_text}"


def chunk_all():
    chunks = []
    for row in read_manifest():
        path = config.PDF_DIR / row["filename"]
        meta = {"doc_id": file_hash(path), "issue": row["issue"], "date": row["date"], "lang": row["lang"]}
        chunks += chunk_document(read_pages(path), meta)
    return chunks
