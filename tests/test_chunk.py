import re

from jarida_rag import config
from jarida_rag.pdf import read_manifest, read_toc


def test_chunks_are_well_formed(chunks):
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["page_start"] <= chunk["page_end"] for chunk in chunks)
    assert all(chunk["text_type"] for chunk in chunks)


def test_every_text_in_the_table_of_contents_is_found(chunks):
    # the printed table of contents is a free answer key for the text detection
    for row in read_manifest():
        expected = set(re.findall(r"رقم\s*(\d[\d.]*\d)\s+صادر", read_toc(config.PDF_DIR / row["filename"])))
        found = {chunk["text_number"] for chunk in chunks if chunk["issue"] == row["issue"]}
        assert expected <= found, f"BO {row['issue']} misses {sorted(expected - found)}"


def test_amendments_are_read_from_the_title(chunks):
    decree = next(chunk for chunk in chunks if chunk["text_number"] == "2.25.761")
    assert decree["amends"] == "2.11.89"
