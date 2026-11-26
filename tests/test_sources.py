from jarida_rag.sources import site


def test_dates_are_read_in_moroccan_time_before_and_after_1970():
    assert site.parse_date("/Date(1767222000000)/") == "2026-01-01"
    assert site.parse_date("/Date(-1798765200000)/") == "1913-01-01"


def test_issues_get_their_storage_path_and_duplicates_are_dropped():
    row = {"BoUrl": "/BO/bo_ar/2026/BO_7470_Ar.pdf", "BoNum": " 7470 ", "BoDate": "/Date(1767222000000)/"}
    assert site.parse_issues("general", [row, row, {"BoUrl": ""}]) == [{
        "url": "https://www.sgg.gov.ma/BO/bo_ar/2026/BO_7470_Ar.pdf",
        "module": "general",
        "issue": "7470",
        "date": "2026-01-01",
        "filename": "BO_7470_Ar.pdf",
        "path": "general/2026/BO_7470_Ar.pdf",
    }]


def test_alternate_url_swaps_the_two_arabic_folders():
    assert site.alternate_url("https://x/BO/Ar/a.pdf") == "https://x/BO/bo_ar/a.pdf"
    assert site.alternate_url("https://x/BO/bo_ar/a.pdf") == "https://x/BO/Ar/a.pdf"
    assert site.alternate_url("https://x/other/a.pdf") is None
