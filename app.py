# app.py
import os
import json
import base64
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

# Only load .env locally. On Railway, use Variables.
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()

# LINE SDK v3
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# Google Gen AI SDK
from google import genai
from google.genai import errors as genai_errors


# =====================================================
# 時區設定（台灣）
# =====================================================
TZ_TAIPEI = timezone(timedelta(hours=8))


# =====================================================
# Env vars
# =====================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GitHub（用 Contents API 寫入 journal，避免 Railway 兩個 service 檔案系統不共享）
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "SonyaSung/sonya-linebot"
GITHUB_PAT = os.getenv("GITHUB_PAT")    # fine-grained PAT with contents write

# Railway 建議設 DATA_DIR=/data（本機可用 data）
DATA_DIR = os.getenv("DATA_DIR", "data")

# Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _present(v: str | None) -> bool:
    return bool(v and v.strip())


missing = []
if not _present(LINE_CHANNEL_ACCESS_TOKEN):
    missing.append("LINE_CHANNEL_ACCESS_TOKEN")
if not _present(LINE_CHANNEL_SECRET):
    missing.append("LINE_CHANNEL_SECRET")
if not _present(GEMINI_API_KEY):
    missing.append("GEMINI_API_KEY")
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


# =====================================================
# App init
# =====================================================
app = FastAPI()

handler = WebhookHandler(LINE_CHANNEL_SECRET)

line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_api_client = ApiClient(line_config)
messaging_api = MessagingApi(line_api_client)

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================
# Helpers: date/time
# =====================================================
def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


def today_ymd() -> str:
    return now_taipei().strftime("%Y-%m-%d")


def now_hm() -> str:
    return now_taipei().strftime("%H:%M")


# =====================================================
# 本機(同一 container)訊息儲存（保留，方便除錯）
# 注意：Railway 的不同 service 不共享檔案系統，所以不要依賴它做跨 service 傳遞
# =====================================================
def log_line_message_local(event: MessageEvent, user_text: str):
    """
    DATA_DIR/messages/YYYY-MM-DD.jsonl
    每行一個 JSON
    """
    try:
        ymd = today_ymd()
        base_dir = Path(DATA_DIR)
        msg_dir = base_dir / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)

        filepath = msg_dir / f"{ymd}.jsonl"

        source = event.source
        chat_type = getattr(source, "type", "unknown")

        record = {
            "timestamp": now_taipei().isoformat(),
            "date": ymd,
            "chat_type": chat_type,
            "user_id": getattr(source, "user_id", None),
            "group_id": getattr(source, "group_id", None),
            "room_id": getattr(source, "room_id", None),
            "message": user_text,
        }

        with filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"log_line_message_local failed: {type(e).__name__}: {e}")


# =====================================================
# GitHub Contents API：直接把對話 append 到 repo 的 journal/YYYY-MM-DD.md
# =====================================================
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
    """
    returns (text_content_or_none, sha_or_none)
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}"
    try:
        data = gh_request("GET", url)
        if not data:
            return None, None
        content_b64 = data.get("content")
        sha = data.get("sha")
        if not content_b64:
            return "", sha
        # GitHub 會用換行分段 base64
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
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    gh_request("PUT", url, payload)


def append_to_daily_journal(lines_to_append: list[str]):
    """
    在 repo 的 journal/YYYY-MM-DD.md 追加內容（不存在就建立）
    """
    if not github_enabled():
        # 沒有設 GitHub 變數也不算錯，只是 journal 不會寫進 repo
        print("GitHub not configured; skip journal upload.")
        return

    ymd = today_ymd()
    path_in_repo = f"journal/{ymd}.md"

    existing, sha = gh_get_file(path_in_repo)

    if existing is None:
        # 新檔：加標題
        base = [f"# {ymd}", ""]
        new_text = "\n".join(base + lines_to_append).rstrip() + "\n"
        gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}")
        return

    # 舊檔：直接 append（確保前面有換行）
    existing_text = existing
    if not existing_text.endswith("\n"):
        existing_text += "\n"
    new_text = (existing_text + "\n".join(lines_to_append).rstrip() + "\n")
    gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}", sha=sha)


# =====================================================
# Gemini 回覆
# =====================================================
def gemini_reply(user_text: str) -> str:
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_text,
        )
        text = (resp.text or "").strip()
        return text if text else "我有收到你的訊息，但模型沒有回傳內容。"

    except genai_errors.APIError as e:
        return f"AI 呼叫失敗：{e.code} {e.message}"
    except Exception as e:
        return f"AI 呼叫失敗：{type(e).__name__}: {e}"


# =====================================================
# Routes
# =====================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "service": "Sonya LINE Bot"}


@app.post("/line/webhook")
async def line_webhook(
    request: Request,
    x_line_signature: str | None = Header(default=None, alias="X-Line-Signature"),
):
    if not x_line_signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body = await request.body()
    body_text = body.decode("utf-8")

    try:
        handler.handle(body_text, x_line_signature)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

    return {"ok": True}


# =====================================================
# LINE event handlers
# =====================================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = (event.message.text or "").strip()

    # 1) 本機記錄（可留可不留）
    log_line_message_local(event, user_text)

    # 2) 先寫入 GitHub journal（使用者訊息）
    try:
        append_to_daily_journal([
            "## LINE 對話",
            f"- [{now_hm()}] Sonya: {user_text}",
        ])
    except Exception as e:
        print(f"append user to journal failed: {type(e).__name__}: {e}")

    # help 指令
    if user_text.lower() in ["/help", "help"]:
        reply_text = (
            "Sonya 生活庶務助手已啟動。\n"
            "\n"
            "目前功能：\n"
            "• Gemini AI 對話\n"
            "• 每則訊息直接寫入 GitHub journal\n"
            "\n"
            "指令：\n"
            "• /help 顯示說明\n"
        )
    else:
        reply_text = gemini_reply(user_text)

    # 3) 回覆給 LINE
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )
    except Exception as e:
        print(f"Reply failed: {type(e).__name__}: {e}")

    # 4) 再把機器人回覆也寫入 GitHub journal
    try:
        append_to_daily_journal([
            f"- [{now_hm()}] Bot: {reply_text}",
            "",
        ])
    except Exception as e:
        print(f"append bot to journal failed: {type(e).__name__}: {e}")
