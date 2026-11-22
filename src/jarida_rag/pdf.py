import csv
import hashlib
import re

import pypdfium2 as pdfium

from . import config

ARABIC_LETTERS = "ء-ي"
ARABIC_LETTER = re.compile(f"[{ARABIC_LETTERS}]")
# decorative stretching ("قــرار"), it breaks title matching
TATWEEL = "ـ"
# pdfium splits the "ال" ligature: "المالية" comes out as "املالية"
BROKEN_AL = re.compile(rf"(?<![{ARABIC_LETTERS}])([وبف]?)ا([{ARABIC_LETTERS}])ل")
DOT_LEADER = re.compile(r"[.…](?:\s?[.…]){2,}")
UNDERSCORES = re.compile(r"_{3,}")
SPACES = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")
TOC_MARKER = re.compile(r"صفحة\s+صفحة")
# pdfium returns Arabic text with "(" and ")" swapped
SWAP_PARENTHESES = str.maketrans("()", ")(")


def read_manifest():
    with open(config.MANIFEST, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_text(text):
    text = text.replace("\r\n", "\n").replace(TATWEEL, "")
    text = BROKEN_AL.sub(r"\1ال\2", text)
    text = text.translate(SWAP_PARENTHESES)
    text = DOT_LEADER.sub(" ", text)
    text = UNDERSCORES.sub(" ", text)
    text = SPACES.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def arabic_ratio(text):
    characters = [char for char in text if not char.isspace()]
    if not characters:
        return 0
    return len(ARABIC_LETTER.findall(text)) / len(characters)


def read_pages(path, issue):
    pdf = pdfium.PdfDocument(path)
    pages = []
    skipped = []
    for index in range(len(pdf)):
        number = index + 1
        text = clean_text(pdf[index].get_textpage().get_text_range())
        if TOC_MARKER.search(text):
            continue
        if arabic_ratio(text) < config.MIN_ARABIC_RATIO:
            skipped.append(number)
            continue
        lines = [line for line in text.split("\n") if not ("الجريدة الرسمية" in line and issue in line)]
        pages.append({"page": number, "text": "\n".join(lines)})
    pdf.close()

    if skipped:
        print(f"{path.name}: skipped pages with too little Arabic text: {skipped}")
    return pages
