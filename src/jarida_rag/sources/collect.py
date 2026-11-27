import asyncio
import hashlib

from curl_cffi.requests import AsyncSession, RequestsError

from jarida_rag import config
from jarida_rag.sources import catalog, drive, site


def collect(limit=10):
    """List every issue on the site, then store the `limit` newest ones not yet stored."""
    with catalog.connect() as db:
        catalog.save_listed(db, site.list_issues())
        results = asyncio.run(store_all(catalog.pending(db, limit)))
        catalog.save_results(db, results)
        return catalog.counts(db)


async def store_all(documents):
    slots = asyncio.Semaphore(config.PARALLEL_DOWNLOADS)
    async with AsyncSession(impersonate="chrome") as session:
        return await asyncio.gather(*(store(session, slots, document) for document in documents))


async def store(session, slots, document):
    async with slots:
        try:
            content = await site.download(session, document["url"])
            drive_id = await asyncio.to_thread(drive.upload, document["path"], content)
        except (RequestsError, RuntimeError) as error:
            result = failed(document, str(error))
        else:
            result = {
                "url": document["url"],
                "status": "stored",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "drive_id": drive_id,
                "error": None,
            }
        await asyncio.sleep(config.REQUEST_PAUSE)
    return result


def failed(document, error):
    return {
        "url": document["url"],
        "status": "failed",
        "size": None,
        "sha256": None,
        "drive_id": None,
        "error": error,
    }


if __name__ == "__main__":
    for row in collect():
        print(row["status"], row["count"])
