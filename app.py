import os
from fastapi import FastAPI, Request, HTTPException
from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from dotenv import load_dotenv
from google import genai

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET or not GEMINI_API_KEY:
    raise RuntimeError("Missing env vars. Please fill LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, GEMINI_API_KEY in .env")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# LINE config
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/line/webhook")
async def line_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True}


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text

    system_prompt = (
        "你是 Sonya 的私人 LINE AI 助理。\n"
        "請一律使用【繁體中文】回答，避免簡體。\n"
        "語氣：直接、少廢話、提供可執行步驟。\n"
    )

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_prompt}\n使用者訊息：{user_text}"
        )

        answer = getattr(resp, "text", None)
        if not answer:
            answer = "（AI沒有回覆內容）"

    except Exception as e:
        answer = f"AI 呼叫失敗：{str(e)}"

    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApi(api_client)
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=answer)]
            )
        )
