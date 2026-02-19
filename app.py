# app.py
import os
import json
import base64
import random
import time
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
- Maximum 1–2 sentences total.

Journal rules:
- Write naturally as Sonya's assistant.
- No meta explanation.

Your personality:
calm, precise, intelligent, minimal.
"""

TRANSLATION_CONCISE_SYSTEM_PROMPT = """
你是一個純翻譯引擎，不是老師、解說者或語言顧問。

當使用者要求翻譯時，必須嚴格遵守以下規則：

━━━━━━━━━━━━━━━━━━
【輸出規則（強制）】
━━━━━━━━━━━━━━━━━━

只輸出最終翻譯結果。

禁止輸出：

- 解釋
- 條列
- 編號
- 多個版本
- 括號註解
- 音標
- Markdown
- 前言
- 結語
- 說明文字
- 「以下是翻譯」
- 「翻譯如下」

輸出只能包含：

→ 一行純翻譯文字

不得包含任何其他內容。

━━━━━━━━━━━━━━━━━━
【性別敬語規則（全語言適用）】
━━━━━━━━━━━━━━━━━━

若翻譯語言涉及性別敬語（例如泰文、日文、韓文、印尼文等）：

一律使用：

→ 女性說話者版本

禁止：

- 輸出男性版本
- 同時輸出男女版本
- 詢問使用者性別
- 解釋性別差異

除非使用者明確指定男性，否則永遠使用女性版本。

━━━━━━━━━━━━━━━━━━
【範例】

使用者：
請翻譯為泰文：今天我吃得很飽，你呢？

正確輸出：
วันนี้ฉันกินอิ่มมากค่ะ คุณล่ะคะ?

錯誤輸出（禁止）：
วันนี้ผมกินอิ่มมากครับ...
วันนี้ฉันกินอิ่มมากค่ะ...
（男性 / 女性版本）

錯誤輸出（禁止）：
以下是翻譯：
วันนี้ฉันกินอิ่มมากค่ะ

━━━━━━━━━━━━━━━━━━

你必須永遠遵守這些規則。
"""

# =====================================================
# 時區設定（台灣）
# =====================================================
TZ_TAIPEI = timezone(timedelta(hours=8))


# =====================================================
# Build / identity (for debugging what code is running)
# =====================================================
APP_NAME = "sonya-linebot"
APP_FILE = __file__
# Railway 常見的 git sha env（不保證一定存在，所以用多個 fallback）
APP_BUILD = (
    os.getenv("RAILWAY_GIT_COMMIT_SHA")
    or os.getenv("GIT_COMMIT_SHA")
    or os.getenv("RAILWAY_COMMIT_SHA")
    or "unknown"
)
APP_ENV = os.getenv("RAILWAY_ENVIRONMENT", os.getenv("ENV", "unknown"))


# =====================================================
# Env vars
# =====================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GitHub（用 Contents API 寫入 journal，避免 Railway 兩個 service 檔案系統不共享）
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "SonyaSung/sonya-linebot"
GITHUB_PAT = os.getenv("GITHUB_PAT")    # fine-grained PAT with contents write

# ⚠️ 建議：journal 另寫到 journal branch，避免 main 觸發 redeploy
JOURNAL_BRANCH = os.getenv("JOURNAL_BRANCH", "main")  # 建議改成 "journal"

# Railway 建議設 DATA_DIR=/data（本機可用 data）
DATA_DIR = os.getenv("DATA_DIR", "data")

# Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL_1 = os.getenv("GEMINI_FALLBACK_MODEL_1")
GEMINI_FALLBACK_MODEL_2 = os.getenv("GEMINI_FALLBACK_MODEL_2")

AI_BUSY_MESSAGE = "現在 AI 服務忙碌，我已收到你的訊息，請稍後再試一次。"

# 觸發字：群組/聊天室只在看到這些才回
TRIGGERS = ["@宋家萬事興", "/bot"]


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

# Print identity at startup (shows in Railway deploy logs)
print(f"[{APP_NAME}] booting...")
print(f"[{APP_NAME}] file={APP_FILE}")
print(f"[{APP_NAME}] build={APP_BUILD} env={APP_ENV}")
print(f"[{APP_NAME}] journal_branch={JOURNAL_BRANCH}")

# Minimal request logging for Railway HTTP logs
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_ts = datetime.now(timezone.utc)
    response = await call_next(request)
    dur_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
    print(
        f'HTTP {request.method} {request.url.path} -> {response.status_code} {dur_ms}ms'
    )
    return response

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
    """
    在 repo 的 journal/YYYY-MM-DD.md 追加內容（不存在就建立）
    """
    if not github_enabled():
        print("GitHub not configured; skip journal upload.")
        return

    ymd = today_ymd()
    path_in_repo = f"journal/{ymd}.md"

    existing, sha = gh_get_file(path_in_repo)

    if existing is None:
        base = [f"# {ymd}", "", "## LINE 對話", ""]
        new_text = "\n".join(base + lines_to_append).rstrip() + "\n"
        gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}")
        return

    existing_text = existing
    if not existing_text.endswith("\n"):
        existing_text += "\n"
    new_text = (existing_text + "\n".join(lines_to_append).rstrip() + "\n")
    gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}", sha=sha)


# =====================================================
# Gemini 回覆
# =====================================================
def _is_retryable_error(err: Exception) -> bool:
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, genai_errors.APIError):
        return err.code in (429, 503)
    return False


def call_gemini(
    prompt: str,
    system_instruction: str | None = None,
    generation_config: dict | None = None,
) -> str | None:
    models = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL_1, GEMINI_FALLBACK_MODEL_2]
    models = [m for m in models if m]

    backoffs = [0.8, 1.6, 3.2, 6.4]
    last_error: Exception | None = None

    for model in models:
        for delay in backoffs:
            try:
                resp = genai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                if not _is_retryable_error(e):
                    break
                jitter = random.uniform(0.0, 0.2)
                time.sleep(delay + jitter)
        # try next fallback model
        continue

    if last_error:
        print(f"Gemini call failed: {type(last_error).__name__}: {last_error}")
    return None


def gemini_reply(user_text: str) -> str:
    full_prompt = SYSTEM_PROMPT + "\n\nUser message:\n" + user_text
    text = call_gemini(full_prompt)
    if text:
        return text
    return AI_BUSY_MESSAGE


def gemini_translate_reply(user_text: str) -> str:
    text = call_gemini(
        user_text,
        system_instruction=TRANSLATION_CONCISE_SYSTEM_PROMPT,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 80,
        },
    )
    if text:
        return text
    return AI_BUSY_MESSAGE


# =====================================================
# Routes
# =====================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "service": "Sonya LINE Bot", "build": APP_BUILD}


@app.get("/__whoami")
def whoami():
    return {
        "app": APP_NAME,
        "file": APP_FILE,
        "build": APP_BUILD,
        "env": APP_ENV,
        "journal_branch": JOURNAL_BRANCH,
    }


@app.get("/routes")
def routes():
    # 顯示目前 FastAPI 註冊的所有路由（debug 用）
    out = []
    for r in app.router.routes:
        methods = sorted(getattr(r, "methods", []) or [])
        path = getattr(r, "path", "")
        name = getattr(r, "name", "")
        out.append({"methods": methods, "path": path, "name": name})
    return {"count": len(out), "routes": out}


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
    is_translation = "翻譯" in user_text

    source = event.source
    chat_type = getattr(source, "type", "unknown")  # user / group / room

    # 1) 本機記錄（可留可不留）
    log_line_message_local(event, user_text)

    # 2) 先寫入 GitHub journal（使用者訊息，包含來源）
    try:
        append_to_daily_journal([
            f"- [{now_hm()}] ({chat_type}) Sonya: {user_text}",
        ])
    except Exception as e:
        print(f"append user to journal failed: {type(e).__name__}: {e}")

    # 3) 回覆策略：群組/聊天室只有觸發才回
    should_reply = True
    if chat_type in ("group", "room"):
        should_reply = any(t in user_text for t in TRIGGERS)

    if not should_reply:
        return

    # 4) help 指令
    if user_text.lower() in ["/help", "help"]:
        reply_text = (
            "宋家萬事興已啟動。\n"
            "\n"
            "規則：\n"
            "• 私訊：每句回覆\n"
            "• 群組/聊天室：只有叫我（@宋家萬事興 或 /bot）才回\n"
            "\n"
            "指令：\n"
            "• /help 顯示說明\n"
        )
    else:
        # 群組/聊天室：去掉觸發字再送 Gemini
        cleaned = user_text
        if chat_type in ("group", "room"):
            for t in TRIGGERS:
                cleaned = cleaned.replace(t, "").strip()
        prompt = cleaned if cleaned else user_text
        if is_translation:
            reply_text = gemini_translate_reply(prompt)
        else:
            reply_text = gemini_reply(prompt)

    # 5) 回覆給 LINE
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )
    except Exception as e:
        print(f"Reply failed: {type(e).__name__}: {e}")

    # 6) 再把機器人回覆也寫入 GitHub journal
    try:
        append_to_daily_journal([
            f"- [{now_hm()}] Bot: {reply_text}",
            "",
        ])
    except Exception as e:
        print(f"append bot to journal failed: {type(e).__name__}: {e}")
