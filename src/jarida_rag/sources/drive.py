import json
import subprocess

from jarida_rag import config


def upload(path, content):
    target = f"{config.REMOTE}/{path}"
    rclone("rcat", target, content=content)
    if json.loads(rclone("lsjson", "--stat", target))["Size"] != len(content):
        raise RuntimeError(f"{target} is incomplete in storage")


def rclone(*args, content=None):
    try:
        result = subprocess.run(["rclone", *args], input=content, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rclone {args[0]} timed out") from None
    if result.returncode != 0:
        raise RuntimeError(f"rclone {args[0]}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout
