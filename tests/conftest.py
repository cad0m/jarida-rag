import pytest

from jarida_rag import config
from jarida_rag.chunk import chunk_all
from jarida_rag.pdf import read_manifest


@pytest.fixture(scope="session")
def chunks():
    # the PDFs aren't in git: tests that need them are skipped instead of failing
    if not all((config.PDF_DIR / row["filename"]).exists() for row in read_manifest()):
        pytest.skip("gazette PDFs are not in data/")
    return chunk_all()
