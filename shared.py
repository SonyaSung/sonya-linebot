# shared.py
import os
import re
import json
import time
import base64
import random
from datetime import datetime, timezone, timedelta
from urllib import request as urlrequest
from urllib.parse import quote
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
from google import genai
import requests
import hmac
import hashlib
from linebot.v3.messaging import ApiClient, Configuration, MessagingApi

if os.path.exists(".env"):
    load_dotenv()

SYSTEM_PROMPT = """
You are Sonya Life Assistant.

Core behavior rules:
- Always respond concisely.
- Default response length: one sentence.
- Never provide explanations unless explicitly requested.
- Never provide multiple alternative answers.
- Never say phrases like "Here are several translations" or "As an AI".
- Never include numbered options unless asked.

Translation rules:
- Output ONLY the translated sentence.
- No explanation.
- No alternatives.

Answer rules:
- Provide direct answer first.
- Maximum 1-2 sentences total.

Journal rules:
- Write naturally as Sonya's assistant.
- No meta explanation.

Your personality:
calm, precise, intelligent, minimal.
"""

TRANSLATION_CONCISE_SYSTEM_PROMPT = """
你是專業翻譯引擎。
你的任務只有一件事：把輸入內容翻譯成目標語言。

輸出規則：
- 只輸出最終翻譯結果。
- 不要任何解釋。
- 不要提供多個版本或選項。
- 不要加標題、前言、結語。
- 不要加編號、符號、Markdown。
- 不要重述原文。

若輸入是片語、短句或單字，也只回覆對應翻譯。
"""

INDONESIAN_TRANSLATION_SYSTEM_PROMPT = (
    "You are a translation engine. Output ONLY the final Indonesian translation. "
    "No explanations, no options, no labels, no markdown, no extra lines."
)

TZ_TAIPEI = timezone(timedelta(hours=8))

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL_1 = os.getenv("GEMINI_FALLBACK_MODEL_1")
GEMINI_FALLBACK_MODEL_2 = os.getenv("GEMINI_FALLBACK_MODEL_2")

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_PAT = os.getenv("GITHUB_PAT")
JOURNAL_BRANCH = os.getenv("JOURNAL_BRANCH", "main")

AI_BUSY_GENERAL_MESSAGE = "Now I'm a bit busy, please try again shortly."
AI_BUSY_TRANSLATION_MESSAGE = "Translation is busy now, please try again shortly."
AI_BUSY_MESSAGE = AI_BUSY_GENERAL_MESSAGE


def _present(v: str | None) -> bool:
    return bool(v and v.strip())


shared_missing = []
if not _present(LINE_CHANNEL_ACCESS_TOKEN):
    shared_missing.append("LINE_CHANNEL_ACCESS_TOKEN")
if not _present(GEMINI_API_KEY):
    shared_missing.append("GEMINI_API_KEY")
if shared_missing:
    raise RuntimeError(f"Missing env vars: {', '.join(shared_missing)}")


line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_api_client = ApiClient(line_config)
messaging_api = MessagingApi(line_api_client)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


def today_ymd() -> str:
    return now_taipei().strftime("%Y-%m-%d")


def now_hm() -> str:
    return now_taipei().strftime("%H:%M")


def github_enabled() -> bool:
    return _present(GITHUB_REPO) and _present(GITHUB_PAT)


def gh_request(method: str, url: str, payload: dict | None = None) -> dict | None:
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "sonya-linebot",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        print(f"GitHub API HTTPError {e.code} on {url}: {body}")
        raise
    except URLError as e:
        print(f"GitHub API URLError on {url}: {e}")
        raise


def gh_get_file(path_in_repo: str) -> tuple[str | None, str | None]:
    ref = quote(JOURNAL_BRANCH, safe="")
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}?ref={ref}"
    try:
        data = gh_request("GET", url)
        if not data:
            return None, None
        content_b64 = data.get("content")
        sha = data.get("sha")
        if not content_b64:
            return "", sha
        content_b64 = content_b64.replace("\n", "")
        text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        return text, sha
    except HTTPError as e:
        if e.code == 404:
            return None, None
        raise


def gh_put_file(path_in_repo: str, text: str, message: str, sha: str | None = None):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}"
    content_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    payload = {
        "message": message,
        "content": content_b64,
        "branch": JOURNAL_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    gh_request("PUT", url, payload)


def append_to_daily_journal(lines_to_append: list[str]):
    if not github_enabled():
        print("GitHub not configured; skip journal upload.")
        return

    ymd = today_ymd()
    path_in_repo = f"journal/{ymd}.md"

    existing, sha = gh_get_file(path_in_repo)

    if existing is None:
        base = [f"# {ymd}", "", "## LINE Log", ""]
        new_text = "\n".join(base + lines_to_append).rstrip() + "\n"
        gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}")
        return

    existing_text = existing
    if not existing_text.endswith("\n"):
        existing_text += "\n"
    new_text = existing_text + "\n".join(lines_to_append).rstrip() + "\n"
    gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}", sha=sha)


def call_gemini(prompt: str, system_instruction: str | None = None) -> str | None:
    contents = prompt

    if system_instruction:
        contents = system_instruction + "\n\n" + prompt

    try:
        # Try modern SDK with GenerateContentConfig
        try:
            from google.genai.types import GenerateContentConfig
            config = GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=4096,
            )
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )
        except (ImportError, TypeError):
            # Fall back to older SDK or simpler interface
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )

        return resp.text if resp and resp.text else "No response from AI."

    except Exception as e:
        print(f"GEMINI_FAIL {type(e).__name__}:{e}", flush=True)
        return "AI busy, please try again."


def _line_headers():
    return {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def line_reply_request(reply_token: str, messages: list[str], timeout=(3, 5)):
    url = "https://api.line.me/v2/bot/message/reply"
    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": t} for t in messages],
    }
    return requests.post(url, json=body, headers=_line_headers(), timeout=timeout)


def line_push_request(to: str, messages: list[str], timeout=(3, 10)):
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": to, "messages": [{"type": "text", "text": t} for t in messages]}
    return requests.post(url, json=body, headers=_line_headers(), timeout=timeout)


def _is_keep_mode(text: str) -> bool:
    return (
        re.match(r"^\s*[#\/]?(keep|id)\b", text, re.IGNORECASE) is not None
        or re.match(r"^\s*(收錄|記錄)\b", text) is not None
    )


def _match_id_translation_single(text: str) -> str | None:
    m = re.match(r"^\s*翻譯成印尼文[:：]?\s*(.*)$", text)
    return m.group(1).strip() if m else None


def _match_id_translation_three(text: str) -> str | None:
    m = re.match(r"^\s*翻譯成印尼文三種版本[:：]?\s*(.*)$", text)
    return m.group(1).strip() if m else None


def _build_id_translation_prompt(source_text: str, variants: int) -> str:
    if variants == 1:
        return (
            "Please translate the following text to Indonesian and output one final sentence only."
            "\nNo explanation, no labels, no markdown."
            f"\nSource: {source_text}"
        )
    return (
        "Please translate the following text to Indonesian in three different versions."
        "\nOutput exactly three lines only, without numbering or explanation."
        f"\nSource: {source_text}"
    )


def _first_n_lines(text: str, n: int) -> list[str]:
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    return lines[:n]


def gemini_reply(user_text: str) -> str:
    full_prompt = SYSTEM_PROMPT + "\n\nUser message:\n" + user_text
    text = call_gemini(full_prompt)
    if text:
        return text
    return AI_BUSY_MESSAGE
