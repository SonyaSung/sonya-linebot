# app.py
import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

# Railway 建議設 DATA_DIR=/data
DATA_DIR = os.getenv("DATA_DIR", "data")

# Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _present(v: str | None) -> bool:
    return bool(v and v.strip())


if not (_present(LINE_CHANNEL_ACCESS_TOKEN) and _present(LINE_CHANNEL_SECRET) and _present(GEMINI_API_KEY)):
    missing = []
    if not _present(LINE_CHANNEL_ACCESS_TOKEN):
        missing.append("LINE_CHANNEL_ACCESS_TOKEN")
    if not _present(LINE_CHANNEL_SECRET):
        missing.append("LINE_CHANNEL_SECRET")
    if not _present(GEMINI_API_KEY):
        missing.append("GEMINI_API_KEY")
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
# 訊息儲存功能
# =====================================================
def today_ymd() -> str:
    """取得今天日期 YYYY-MM-DD"""
    return datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d")


def log_line_message(event: MessageEvent, user_text: str):
    """
    將 LINE 訊息存入
    DATA_DIR/messages/YYYY-MM-DD.jsonl

    jsonl 格式：每行一個 JSON
    """

    try:
        ymd = today_ymd()

        base_dir = Path(DATA_DIR)
        msg_dir = base_dir / "messages"

        msg_dir.mkdir(parents=True, exist_ok=True)

        filepath = msg_dir / f"{ymd}.jsonl"

        # 判斷聊天來源
        source = event.source
        chat_type = getattr(source, "type", "unknown")

        record = {
            "timestamp": datetime.now(TZ_TAIPEI).isoformat(),
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
        print(f"log_line_message failed: {type(e).__name__}: {e}")


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

        if text:
            return text
        else:
            return "我有收到你的訊息，但模型沒有回傳內容。"

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
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": str(e)}
        )

    return {"ok": True}


# =====================================================
# LINE event handlers
# =====================================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):

    user_text = (event.message.text or "").strip()

    # ⭐⭐⭐ 關鍵：先儲存訊息 ⭐⭐⭐
    log_line_message(event, user_text)

    # help 指令
    if user_text.lower() in ["/help", "help"]:
        reply_text = (
            "Sonya 生活庶務助手已啟動。\n"
            "\n"
            "目前功能：\n"
            "• Gemini AI 對話\n"
            "• 自動記錄所有 LINE 訊息\n"
            "\n"
            "未來功能：\n"
            "• 每日自動產生日記\n"
            "• 翻譯印尼語\n"
            "• 黃金報價\n"
            "• 番茄鐘"
        )
    else:
        reply_text = gemini_reply(user_text)

    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )

    except Exception as e:
        print(f"Reply failed: {type(e).__name__}: {e}")
