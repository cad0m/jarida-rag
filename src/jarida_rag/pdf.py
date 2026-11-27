"""Turn a gazette PDF into clean page texts.

pdfium extracts Arabic text well, but with known defects that are repaired here:
broken "ال" and "لا" ligatures, mirrored brackets, reversed Latin words, and the running page header.
"""
import csv
import hashlib
import re
import statistics
import unicodedata

import pypdfium2 as pdfium
from wordfreq import zipf_frequency

from . import config

ARABIC = "ء-ي"
ARABIC_LETTER = re.compile(f"[{ARABIC}]")
ARABIC_WORD = re.compile(f"[{ARABIC}]+")
LATIN_RUN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9’'.,&\- ]*[A-Za-zÀ-ÿ0-9]")
# a line with no letter, digit or "*" is layout debris: lone diacritics, dots, rules
# ("* * *" lines separate short notices, so they stay)
DEBRIS_LINE = re.compile(r"[^\w*]*")
TOC_MARKER = re.compile(r"صفحة\s+صفحة")

# "المالية" comes out as "املالية": the article's lam lands after the next letter
BROKEN_AL = re.compile(rf"(?<![{ARABIC}])([وبف]?)ا([{ARABIC}])ل")
# the lam-alef ligature "لا" comes out as "ال", which is also the article, so a dictionary decides
ALEF_LAM = re.compile("([اأإآ])ل")
STANDALONE_LA = {"ال": "لا", "وال": "ولا", "فال": "فلا", "بال": "بلا", "كال": "كلا"}

MIRRORED = str.maketrans("()«»[]", ")(»«][")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\ufffd\u0640"))  # zero-width, broken glyph, tatweel

DOT_LEADER = re.compile(r"[.…](?:\s?[.…]){2,}")
UNDERSCORES = re.compile(r"_{3,}")
SPACES = re.compile(r"[ \t]+")


def read_manifest():
    with open(config.MANIFEST, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arabic_ratio(text):
    characters = [char for char in text if not char.isspace()]
    return len(ARABIC_LETTER.findall(text)) / len(characters) if characters else 0


# ---------- text repair ----------

def fix_lam_alef(match):
    word = match.group(0)
    if word in STANDALONE_LA:
        return STANDALONE_LA[word]
    # try swapping each "ال" back to "لا", one at a time, and keep a swap only when it turns
    # a rare word into a clearly common one ("إعالن" -> "إعلان"); the article "ال" never passes this test
    best, best_score = word, zipf_frequency(word, "ar") + 1
    for match in ALEF_LAM.finditer(word):
        candidate = word[:match.start()] + "ل" + match.group(1) + word[match.end():]
        score = zipf_frequency(candidate, "ar")
        if score > best_score:
            best, best_score = candidate, score
    return best


def fix_latin_runs(line):
    # Latin words inside right-to-left text come out in reverse order: "Bank Attijariwafa" -> "Attijariwafa Bank"
    def reverse(match):
        parts = re.findall(r"\w+|\W+", match.group(0))
        return "".join(part if part[0].isalnum() else part[::-1] for part in reversed(parts))
    return LATIN_RUN.sub(reverse, line)


def clean_line(line):
    line = BROKEN_AL.sub(r"\1ال\2", line)
    line = ARABIC_WORD.sub(fix_lam_alef, line)
    line = fix_latin_runs(line)
    line = DOT_LEADER.sub(" ", line)
    line = UNDERSCORES.sub(" ", line)
    return SPACES.sub(" ", line).strip()


def clean_text(text):
    text = unicodedata.normalize("NFKC", text).translate(INVISIBLE)
    text = text.translate(MIRRORED).translate(ARABIC_DIGITS).replace("هللا", "الله")
    lines = (clean_line(line) for line in text.replace("\r\n", "\n").split("\n"))
    return "\n".join(line for line in lines if not DEBRIS_LINE.fullmatch(line))


# ---------- pages ----------

def page_body(page):
    # the running header sits in a fixed band at the top: crop it by position, whatever its text looks like
    width, height = page.get_size()
    return page.get_textpage().get_text_bounded(0, 0, width, config.HEADER_LINE * height)


def page_status(text):
    if TOC_MARKER.search(text):
        return "toc"
    if len(text) < config.MIN_PAGE_CHARS:
        return "no_text"  # scan, map or figure
    if arabic_ratio(text) < config.MIN_ARABIC_RATIO:
        return "low_arabic"  # tables or French: kept, but reported
    return "ok"


def read_pages(source, name):
    """Clean text of every indexable page of a PDF (a path or its bytes).

    Skipped and suspicious pages are reported, never dropped silently.
    """
    pdf = pdfium.PdfDocument(source)
    pages = []
    for index in range(len(pdf)):
        text = clean_text(page_body(pdf[index]))
        pages.append({"page": index + 1, "text": text, "status": page_status(text)})
    pdf.close()

    flagged = {}
    for page in pages:
        if page["status"] != "ok":
            flagged.setdefault(page["status"], []).append(page["page"])
    print(f"{name}: {len(pages)} pages, flagged {flagged}")

    # a whole issue in an old non-Unicode font (or a scan) is unreadable: skip it instead of indexing garbage
    if not pages or statistics.median(arabic_ratio(page["text"]) for page in pages) < config.MIN_DOCUMENT_ARABIC_RATIO:
        print(f"{name}: SKIPPED, the text layer is not readable Arabic (needs OCR)")
        return []
    return [page for page in pages if page["status"] in ("ok", "low_arabic")]


def read_toc(path):
    pdf = pdfium.PdfDocument(path)
    texts = [clean_text(page_body(pdf[index])) for index in range(len(pdf))]
    pdf.close()
    return "\n".join(text for text in texts if TOC_MARKER.search(text))
