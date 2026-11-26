from jarida_rag.evaluation.scoring import has_key_facts

# each case is one the scorer once got wrong


def test_yes_no_is_case_insensitive_and_a_whole_word():
    assert has_key_facts("Yes, they are considered present.", "yes")
    assert not has_key_facts("Yes, the meetings are public, not closed.", "no")


def test_arabic_article_merged_with_a_preposition():
    assert has_key_facts("ممثل عن المديرية الإقليمية للتجهيز", "التجهيز")


def test_numbers_written_in_words_or_inside_other_numbers():
    assert has_key_facts("valid for two years", "2")
    assert not has_key_facts("decree 2.26.28 sets it", "2")


def test_citation_labels_are_not_facts():
    assert not has_key_facts("see [BO 7470, 2.24.830, المادة 5, p. 8]", "5")


def test_and_or_parts():
    assert has_key_facts("within sixty (60) days", "60&days|months")
    assert not has_key_facts("within sixty (60) days", "60&weeks")
