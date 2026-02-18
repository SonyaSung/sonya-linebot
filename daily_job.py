import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))

DATA_DIR = os.getenv("DATA_DIR", "data")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_PAT = os.getenv("GITHUB_PAT")

NOTES_DIR = Path(DATA_DIR) / "notes_repo"


def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def today():
    return datetime.now(TZ).strftime("%Y-%m-%d")


def load_messages(date):
    fp = Path(DATA_DIR) / "messages" / f"{date}.jsonl"
    if not fp.exists():
        return []

    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_md(date, rows):
    lines = [f"# {date}", "", "## LINE 訊息記錄", ""]

    for r in rows:
        ts = r["timestamp"]
        msg = r["message"]
        lines.append(f"- {ts} {msg}")

    return "\n".join(lines)


def ensure_repo():
    if not NOTES_DIR.exists():
        url = f"https://{GITHUB_PAT}@github.com/{GITHUB_REPO}.git"
        run(f"git clone {url} {NOTES_DIR}")

    run("git pull", cwd=NOTES_DIR)


def write_md(date, content):
    fp = NOTES_DIR / f"{date}.md"
    fp.write_text(content, encoding="utf-8")


def push(date):
    run("git add .", cwd=NOTES_DIR)
    run(f'git commit -m "daily: {date}" || true', cwd=NOTES_DIR)
    run("git push", cwd=NOTES_DIR)


def main():
    date = today()
    ensure_repo()

    rows = load_messages(date)
    md = make_md(date, rows)

    write_md(date, md)
    push(date)


if __name__ == "__main__":
    main()
