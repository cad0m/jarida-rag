import asyncio
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote, urljoin, urlsplit

from curl_cffi.requests import RequestsError, Session

SITE = "https://www.sgg.gov.ma/"
PAGE_URL = SITE + "arabe/BulletinOfficiel.aspx"
LIST_URL = SITE + "arabe/DesktopModules/MVC/TableListBO/BO/AjaxMethod"
TAB_ID = "847"
MODULES = {"3111": "general", "3110": "agreements"}

TOKEN = re.compile(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"')
DOTNET_DATE = re.compile(r"/Date\((-?\d+)")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MOROCCO = timezone(timedelta(hours=1))
RETRIES = 3


def list_issues():
    with Session(impersonate="chrome") as session:
        page = session.get(PAGE_URL, timeout=60)
        page.raise_for_status()
        token = TOKEN.search(page.text)
        if not token:
            raise RuntimeError("no __RequestVerificationToken on the gazette page")
        return [
            issue
            for module_id, module in MODULES.items()
            for issue in parse_issues(module, list_module(session, module_id, token.group(1)))
        ]


def list_module(session, module_id, token):
    response = session.get(LIST_URL, timeout=120, headers={
        "ModuleId": module_id,
        "TabId": TAB_ID,
        "RequestVerificationToken": token,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": PAGE_URL,
    })
    response.raise_for_status()
    data = response.json()
    return data["data"] if isinstance(data, dict) else data


def parse_issues(module, rows):
    issues = {}
    for row in rows:
        if row.get("BoUrl") and row.get("BoDate"):
            issue = parse_issue(module, row)
            issues[issue["url"]] = issue
    return list(issues.values())


def parse_issue(module, row):
    url = normalize_url(row["BoUrl"])
    date = parse_date(row["BoDate"])
    filename = unquote(url.rsplit("/", 1)[-1])
    return {
        "url": url,
        "module": module,
        "issue": str(row.get("BoNum") or "").strip(),
        "date": date,
        "filename": filename,
        "path": f"{module}/{date[:4]}/{filename}",
    }


def parse_date(value):
    # issues are stamped at midnight Moroccan time: read in UTC, each would land on the day before
    milliseconds = int(DOTNET_DATE.search(value).group(1))
    return (EPOCH + timedelta(milliseconds=milliseconds)).astimezone(MOROCCO).date().isoformat()


def normalize_url(url):
    parts = urlsplit(urljoin(SITE, url.strip()))
    path = quote(re.sub("/{2,}", "/", parts.path), safe="/%:@&=+$,-_.!~*'()")
    return parts._replace(path=path, fragment="").geturl()


def alternate_url(url):
    # some issues moved between these two folders, and the site still lists the old one
    if "/BO/Ar/" in url:
        return url.replace("/BO/Ar/", "/BO/bo_ar/")
    if "/BO/bo_ar/" in url:
        return url.replace("/BO/bo_ar/", "/BO/Ar/")
    return None


async def download(session, url):
    response = await get(session, url)
    if response.status_code == 404 and alternate_url(url):
        url = alternate_url(url)
        response = await get(session, url)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {url}")
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"not a PDF: {url}")
    return response.content


async def get(session, url):
    for attempt in range(1, RETRIES + 1):
        try:
            response = await session.get(url, timeout=120)
            if response.status_code < 500 or attempt == RETRIES:
                return response
        except RequestsError:
            if attempt == RETRIES:
                raise
        await asyncio.sleep(2 ** attempt)
