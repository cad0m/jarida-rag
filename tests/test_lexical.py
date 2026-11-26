from jarida_rag import lexical


def test_tokenize_strips_the_article_but_not_root_letters():
    assert lexical.tokenize("والمرسوم") == ["مرسوم"]
    assert lexical.tokenize("للتجهيز") == ["تجهيز"]
    assert lexical.tokenize("وزير") == ["وزير"]


def test_tokenize_keeps_decree_numbers_whole():
    assert lexical.tokenize("رقم 2.24.830 الصادر") == ["رقم", "2.24.830", "صادر"]


def test_fuse_ranks_a_chunk_found_by_both_retrievers_first():
    assert lexical.fuse([{"id": "a"}, {"id": "b"}], [{"id": "b"}])[0]["id"] == "b"


def test_bm25_finds_a_decree_by_its_number(chunks):
    top = lexical.search(lexical.build(chunks), "ما هو القرار الذي يتمم القرار 2797.95؟", 3)
    assert any("2797.95" in hit["embed_text"] for hit in top)
