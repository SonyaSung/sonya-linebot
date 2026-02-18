import os
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))

DATA_DIR = os.getenv("DATA_DIR", "data")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # "Owner/Repo"
GITHUB_PAT = os.getenv("GITHUB_PAT")    # fine-grained token
JOURNAL_PATH = os.getenv("JOURNAL_PATH", "journal")  # folder in repo, e.g. "journal"

API_BASE = "https://api.github.com"


def today():
    return datetime.now(TZ).strftime("%Y-%m-%d")


def load_messages(date):
    fp = Path(DATA_DIR) / "messages" / f"{date}.jsonl"
    if not fp.exists():
        return []
    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_md(date, rows):
    lines = [f"# {date}", "", "## LINE 訊息記錄", ""]
    for r in rows:
        ts = r.get("timestamp", "")
        msg = r.get("message", "")
        lines.append(f"- {ts} {msg}")
    lines.append("")
    return "\n".join(lines)


def gh_request(method, url, data=None):
    if not GITHUB_PAT:
        raise RuntimeError("Missing env var: GITHUB_PAT")
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "sonya-life-assistant",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {"raw": raw}
        return e.code, payload


def get_existing_sha(repo, path):
    # GET /repos/{owner}/{repo}/contents/{path}
    url = f"{API_BASE}/repos/{repo}/contents/{path}"
    status, payload = gh_request("GET", url)
    if status == 200 and isinstance(payload, dict) and "sha" in payload:
        return payload["sha"]
    if status == 404:
        return None
    raise RuntimeError(f"GitHub API error GET content: {status} {payload}")


def put_file(repo, path, content_text, message):
    b64 = base64.b64encode(content_text.encode("utf-8")).decode("utf-8")
    sha = get_existing_sha(repo, path)

    url = f"{API_BASE}/repos/{repo}/contents/{path}"
    data = {
        "message": message,
        "content": b64,
    }
    if sha:
        data["sha"] = sha

    status, payload = gh_request("PUT", url, data=data)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub API error PUT: {status} {payload}")
    return payload


def main():
    if not GITHUB_REPO:
        raise RuntimeError("Missing env var: GITHUB_REPO (e.g. SonyaSung/sonya-linebot)")

    date = today()
    rows = load_messages(date)
    md = make_md(date, rows)

    # write to repo folder, e.g. journal/2026-02-18.md
    repo_path = f"{JOURNAL_PATH}/{date}.md"
    put_file(
        GITHUB_REPO,
        repo_path,
        md,
        message=f"daily: {date}",
    )
    print(f"Uploaded: {repo_path}")


if __name__ == "__main__":
    main()
