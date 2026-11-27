import json
import subprocess

from curl_cffi import requests

from jarida_rag import config

DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def upload(path, content):
    """Store content at path in Drive and return its Drive file id."""
    target = f"{config.REMOTE}/{path}"
    rclone("rcat", target, content=content)
    stored = json.loads(rclone("lsjson", "--stat", target))
    if stored["Size"] != len(content):
        raise RuntimeError(f"{target} is incomplete in storage")
    return stored["ID"]


def download(drive_id):
    """Read a PDF by its Drive file id; needs no login, only the folder shared as 'anyone with the link'."""
    response = requests.get(DOWNLOAD_URL, params={"id": drive_id, "export": "download"}, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"Drive file {drive_id} is not readable: is the folder shared by link?")
    return response.content


def rclone(*args, content=None):
    try:
        result = subprocess.run(["rclone", *args], input=content, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rclone {args[0]} timed out") from None
    if result.returncode != 0:
        raise RuntimeError(f"rclone {args[0]}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout
