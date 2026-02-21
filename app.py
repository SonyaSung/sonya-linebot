# app.py
import os
import json
import base64
import time
import re
import random
import hmac
import hashlib
import uuid
from urllib.parse import quote
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Only load .env locally. On Railway, use Variables.
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()

# LINE SDK v3
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
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
from redis import Redis
from rq import Queue, Retry
from shared import TRANSLATION_CONCISE_SYSTEM_PROMPT


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
- Maximum 1?? sentences total.

Journal rules:
- Write naturally as Sonya's assistant.
- No meta explanation.

Your personality:
calm, precise, intelligent, minimal.
"""

INDONESIAN_TRANSLATION_SYSTEM_PROMPT = (
    "You are a translation engine. Output ONLY the final Indonesian translation. "
    "No explanations, no options, no labels, no markdown, no extra lines."
)

# =====================================================
# ?????????????????
# =====================================================
TZ_TAIPEI = timezone(timedelta(hours=8))


# =====================================================
# Build / identity (for debugging what code is running)
# =====================================================
APP_NAME = "sonya-linebot"
APP_FILE = __file__
# Railway ??????git sha env????????????????????????????????????????fallback??
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
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("RAILWAY_REDIS_URL")

# GitHub?????Contents API ????潸縐????????journal??????皝弄??Railway ????????service ?????????????????????
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "SonyaSung/sonya-linebot"
GITHUB_PAT = os.getenv("GITHUB_PAT")    # fine-grained PAT with contents write

# ??? ????????????urnal ?????謑黑???journal branch??????皝弄??main ?????redeploy
JOURNAL_BRANCH = os.getenv("JOURNAL_BRANCH", "main")  # ??????? "journal"

# Railway ????????DATA_DIR=/data??????橘??????????data??
DATA_DIR = os.getenv("DATA_DIR", "data")

# Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL_1 = os.getenv("GEMINI_FALLBACK_MODEL_1")
GEMINI_FALLBACK_MODEL_2 = os.getenv("GEMINI_FALLBACK_MODEL_2")

AI_BUSY_GENERAL_MESSAGE = "Now I'm a bit busy, please try again shortly."
AI_BUSY_TRANSLATION_MESSAGE = "Translation is busy now, please try again shortly."
AI_BUSY_MESSAGE = AI_BUSY_GENERAL_MESSAGE

# Trigger words for group/room messages.
TRIGGERS = ["@\u5b8b\u5bb6\u842c\u4e8b\u8208", "/bot"]


def _present(v: str | None) -> bool:
    return bool(v and v.strip())


missing = []
if not _present(LINE_CHANNEL_ACCESS_TOKEN):
    missing.append("LINE_CHANNEL_ACCESS_TOKEN")
if not _present(LINE_CHANNEL_SECRET):
    missing.append("LINE_CHANNEL_SECRET")
if not _present(GEMINI_API_KEY):
    missing.append("GEMINI_API_KEY")
if not _present(REDIS_URL):
    missing.append("REDIS_URL (or RAILWAY_REDIS_URL)")
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


# =====================================================
# App init
# =====================================================
app = FastAPI()

# Print identity at startup (shows in Railway deploy logs)
print(f"[{APP_NAME}] booting...", flush=True)
print(f"[{APP_NAME}] file={APP_FILE}", flush=True)
print(f"[{APP_NAME}] build={APP_BUILD} env={APP_ENV}", flush=True)
print(f"[{APP_NAME}] journal_branch={JOURNAL_BRANCH}", flush=True)
print(f"[sonya-linebot] redis={'SET' if REDIS_URL else 'UNSET'} queue={os.getenv('RQ_QUEUE','line')} build={APP_BUILD}", flush=True)

# Minimal request logging for Railway HTTP logs
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_ts = datetime.now(timezone.utc)
    response = await call_next(request)
    dur_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
    print(
        f'HTTP {request.method} {request.url.path} -> {response.status_code} {dur_ms}ms',
        flush=True
    )
    return response

handler = WebhookHandler(LINE_CHANNEL_SECRET)

line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_api_client = ApiClient(line_config)
messaging_api = MessagingApi(line_api_client)

genai_client = genai.Client(api_key=GEMINI_API_KEY)
redis_conn = Redis.from_url(REDIS_URL)

# Respect RQ queue name from env so app and worker match
RQ_QUEUE = os.getenv("RQ_QUEUE", "line")
line_queue = Queue(RQ_QUEUE, connection=redis_conn)


def _mask_redis_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        # Mask credentials (everything between :// and @)
        return __import__("re").sub(r"://.*@", "://***@", url)
    except Exception:
        return url


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
# ???(??? container)????????????????????????
# ???獢??????????????肄ay ?????service ????????????????蝛遴??????????????????????????????? service ???
# =====================================================
def log_line_message_local(event: MessageEvent, user_text: str):
    """
    DATA_DIR/messages/YYYY-MM-DD.jsonl
    ???????????????JSON
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
# GitHub Contents API????????????????append ??repo ??journal/YYYY-MM-DD.md
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
    """
    ??repo ??journal/YYYY-MM-DD.md ?????????????????????殉狐???????????????雓???
    """
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
    new_text = (existing_text + "\n".join(lines_to_append).rstrip() + "\n")
    gh_put_file(path_in_repo, new_text, message=f"daily: {ymd}", sha=sha)


# =====================================================
# Gemini ???
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


def _is_keep_mode(text: str) -> bool:
    return (
        re.match(r"^\s*[#\/]?(keep|id)\b", text, re.IGNORECASE) is not None
        or re.match(r"^\s*(\u6536\u9304|\u8a18\u9304)\b", text) is not None
    )


def _match_id_translation_single(text: str) -> str | None:
    m = re.match(r"^\s*\u7ffb\u8b6f\u6210\u5370\u5c3c\u6587[:\uff1a]?\s*(.*)$", text)
    return m.group(1).strip() if m else None


def _match_id_translation_three(text: str) -> str | None:
    m = re.match(r"^\s*\u7ffb\u8b6f\u6210\u5370\u5c3c\u6587\u4e09\u7a2e\u7248\u672c[:\uff1a]?\s*(.*)$", text)
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
    # ??????????授? FastAPI ??????????????debug ???
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
    background_tasks: BackgroundTasks = None,
):
    if not x_line_signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body = await request.body()
    body_text = body.decode("utf-8")

    # Verify signature (HMAC-SHA256, base64)
    try:
        sig = base64.b64encode(hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()).decode()
        if not hmac.compare_digest(sig, x_line_signature):
            return JSONResponse(status_code=400, content={"ok": False, "error": "invalid signature"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

    # Parse events and enqueue in background to ensure fast HTTP 200 response
    try:
        data = json.loads(body_text)
        events = data.get("events", [])
    except Exception:
        events = []

    for ev in events:
        trace_id = uuid.uuid4().hex
        event_type = ev.get("type")
        user = ev.get("source", {}).get("userId") or ev.get("source", {}).get("user_id")
        print(f"[line-webhook] RECEIVED trace_id={trace_id} event_type={event_type} user={user}", flush=True)
        if background_tasks is not None:
            background_tasks.add_task(_process_event_in_background, ev, trace_id)

    return {"ok": True}


def _process_event_in_background(ev: dict, trace_id: str):
    # Build payload from raw event dict (only minimal fields)
    try:
        src = ev.get("source", {})
        chat_type = src.get("type", "unknown")
        to_id = src.get("userId") or src.get("user_id") or src.get("groupId") or src.get("roomId") or None
        user_text = ""
        if ev.get("message"):
            user_text = ev["message"].get("text", "")

        payload = {
            "chat_type": chat_type,
            "to_id": to_id,
            "user_text": user_text,
            "timestamp": ev.get("timestamp", int(time.time() * 1000)),
            "source_user_id": src.get("userId") or src.get("user_id"),
            "trace_id": trace_id,
        }

        redis_display = _mask_redis_url(REDIS_URL)
        try:
            redis_conn.ping()
        except Exception as e:
            print(f"[line-webhook] ENQUEUE_FAIL trace_id={trace_id} reason=redis_unreachable err={type(e).__name__}:{e} redis={redis_display} queue={RQ_QUEUE}", flush=True)
            return

        job = line_queue.enqueue(
            "worker.process_line_job",
            payload,
            job_timeout=120,
            result_ttl=0,
            failure_ttl=3600,
        )
        print(f"[line-webhook] ENQUEUE_OK trace_id={trace_id} job_id={getattr(job, 'id', None)}", flush=True)

        # optional fast ACK reply (non-blocking to HTTP response because we're in background)
        try:
            reply_token = ev.get("replyToken")
            if reply_token:
                from shared import line_reply_request
                try:
                    resp = line_reply_request(reply_token, ["收到，處理中"], timeout=(3, 5))
                    if not (200 <= getattr(resp, 'status_code', 0) < 300):
                        print(f"[line-webhook] ACK_FAIL trace_id={trace_id} status={getattr(resp, 'status_code', None)}", flush=True)
                except Exception as e:
                    print(f"[line-webhook] ACK_FAIL trace_id={trace_id} err={type(e).__name__}:{e}", flush=True)
        except Exception:
            pass

    except Exception as e:
        print(f"[line-webhook] ENQUEUE_FAIL trace_id={trace_id} err={type(e).__name__}:{e}", flush=True)


# =====================================================
# LINE event handlers
# =====================================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_text = (event.message.text or "").strip()
    source = event.source
    chat_type = getattr(source, "type", "unknown")  # user / group / room

    to_id = None
    if chat_type == "user":
        to_id = getattr(source, "user_id", None)
    elif chat_type == "group":
        to_id = getattr(source, "group_id", None)
    elif chat_type == "room":
        to_id = getattr(source, "room_id", None)

    should_process = True
    cleaned = user_text
    if chat_type in ("group", "room"):
        should_process = any(t in user_text for t in TRIGGERS)
        for t in TRIGGERS:
            cleaned = cleaned.replace(t, "").strip()

    if not should_process:
        return

    if not to_id:
        print(
            f"enqueue skipped: missing to_id chat_type={chat_type} msg_len={len(user_text)}",
            flush=True
        )
        return

    payload = {
        "chat_type": chat_type,
        "to_id": to_id,
        "user_text": cleaned if cleaned else user_text,
        "timestamp": getattr(event, "timestamp", int(time.time() * 1000)),
        "source_user_id": getattr(source, "user_id", None),
    }
    # build trace_id preference order: message.id -> replyToken[:8]+timestamp -> uuid4[:12]
    trace_id = None
    try:
        trace_id = getattr(event.message, "id", None)
    except Exception:
        trace_id = None

    if not trace_id:
        reply_token = getattr(event, "reply_token", "") or ""
        ts_short = str(payload.get("timestamp", int(time.time() * 1000)))
        if reply_token:
            trace_id = (reply_token[:8] + ts_short)[:12]
        else:
            trace_id = uuid.uuid4().hex[:12]

    payload["trace_id"] = trace_id

    # Minimal redis reachability check
    redis_display = _mask_redis_url(REDIS_URL)

    print(
        f"[line-webhook] RECEIVED trace_id={trace_id} user_id={payload.get('source_user_id')} "
        f"msg_type={payload.get('chat_type')} text_len={len(payload.get('user_text') or '')}",
        flush=True,
    )

    try:
        try:
            # quick ping to surface redis connectivity issues
            redis_conn.ping()
        except Exception as e:
            print(
                f"[line-webhook] ENQUEUE_FAIL trace_id={trace_id} reason=redis_unreachable "
                f"err={type(e).__name__}:{e} redis={redis_display} queue={RQ_QUEUE}",
                flush=True,
            )
            return

        job = line_queue.enqueue(
            "worker.process_line_job",
            payload,
            retry=Retry(max=3, interval=[10, 30, 60]),
            job_id=f"line:{trace_id}",
            job_timeout=180,
            result_ttl=0,
            ttl=600,
            failure_ttl=86400,
        )

        print(
            f"[line-webhook] ENQUEUE_OK trace_id={trace_id} job_id={job.id} "
            f"queue={RQ_QUEUE} redis={redis_display}",
            flush=True,
        )

    except Exception as e:
        print(
            f"[line-webhook] ENQUEUE_FAIL trace_id={trace_id} err={type(e).__name__}:{e} "
            f"redis={redis_display} queue={RQ_QUEUE}",
            flush=True,
        )
        # swallow enqueue errors so webhook returns 200 fast
        return

    # Optional fast ACK so user gets immediate confirmation.
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="\u6536\u5230\uff0c\u6211\u6574\u7406\u4e00\u4e0b\u3002")],
            )
        )
    except Exception as e:
        print(f"ACK reply failed: {type(e).__name__}: {e}", flush=True)

