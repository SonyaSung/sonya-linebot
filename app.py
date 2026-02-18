# app.py
import os
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

# Google Gen AI SDK (google-genai)
from google import genai
from google.genai import errors as genai_errors

# -------------------------
# Env vars
# -------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model: set in Railway Variables if you want; default here
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _present(v: str | None) -> bool:
    return bool(v and v.strip())

if not (_present(LINE_CHANNEL_ACCESS_TOKEN) and _present(LINE_CHANNEL_SECRET) and _present(GEMINI_API_KEY)):
    # Don't print secrets. Only indicate which is missing.
    missing = []
    if not _present(LINE_CHANNEL_ACCESS_TOKEN): missing.append("LINE_CHANNEL_ACCESS_TOKEN")
    if not _present(LINE_CHANNEL_SECRET): missing.append("LINE_CHANNEL_SECRET")
    if not _present(GEMINI_API_KEY): missing.append("GEMINI_API_KEY")
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

# -------------------------
# App init
# -------------------------
app = FastAPI()

handler = WebhookHandler(LINE_CHANNEL_SECRET)

line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_api_client = ApiClient(line_config)
messaging_api = MessagingApi(line_api_client)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Helpers
# -------------------------
def gemini_reply(user_text: str) -> str:
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_text,
        )
        text = (resp.text or "").strip()
        return text if text else "我有收到你的訊息，但這次模型沒有回傳內容。你可以換個問法再試一次。"

    except genai_errors.APIError as e:
        # Typical: wrong model name (404), auth (401), quota (429)...
        return f"AI 呼叫失敗：{e.code} {e.message}"

    except Exception as e:
        return f"AI 呼叫失敗：{type(e).__name__}: {e}"

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "service": "Sonya LINE Bot"}

# LINE Webhook URL: https://<railway-domain>/line/webhook
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

# -------------------------
# LINE event handlers
# -------------------------
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = (event.message.text or "").strip()

    if user_text.lower() in ["/help", "help"]:
        reply_text = (
            "可以直接傳文字給我，我會用 Gemini 回覆。\n"
            "你也可以在 Railway Variables 設定 GEMINI_MODEL（例如 gemini-2.5-flash）。"
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
