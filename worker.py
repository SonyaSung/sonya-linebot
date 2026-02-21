import os
import time
import traceback

from redis import Redis
from rq import Queue, Worker, get_current_job

from linebot.v3.messaging import PushMessageRequest, TextMessage

from shared import (
    AI_BUSY_TRANSLATION_MESSAGE,
    INDONESIAN_TRANSLATION_SYSTEM_PROMPT,
    _build_id_translation_prompt,
    _first_n_lines,
    _is_keep_mode,
    _match_id_translation_single,
    _match_id_translation_three,
    append_to_daily_journal,
    call_gemini,
    gemini_reply,
    line_push_request,
    now_hm,
)


QUEUE_NAME = os.environ.get("RQ_QUEUE", "line")


def _build_reply_text(prompt: str) -> tuple[str, bool]:
    if _is_keep_mode(prompt):
        return "Received.", False

    if prompt.lower() in ["/help", "help"]:
        return (
            "Send any question directly and I will answer concisely."
            "\nIn groups, include a trigger first."
            "\nTranslation command: use the single-translation trigger text."
        ), False

    translation_single = _match_id_translation_single(prompt)
    translation_three = _match_id_translation_three(prompt)
    if translation_single is not None:
        if not translation_single:
            return "Please provide text to translate.", False
        trans_prompt = _build_id_translation_prompt(translation_single, 1)
        text = call_gemini(
            trans_prompt,
            system_instruction=INDONESIAN_TRANSLATION_SYSTEM_PROMPT,
        )
        lines = _first_n_lines(text or "", 1)
        return (lines[0] if lines else AI_BUSY_TRANSLATION_MESSAGE), True

    if translation_three is not None:
        if not translation_three:
            return "Please provide text to translate.", False
        trans_prompt = _build_id_translation_prompt(translation_three, 3)
        text = call_gemini(
            trans_prompt,
            system_instruction=(
                "You are a translation engine. Output EXACTLY three lines, "
                "each line a different Indonesian translation. "
                "No numbering, no bullets, no labels, no extra text."
            ),
        )
        lines = _first_n_lines(text or "", 3)
        if len(lines) == 3:
            return "\n".join(lines), True
        return AI_BUSY_TRANSLATION_MESSAGE, True

    return gemini_reply(prompt), True


def process_line_job(payload: dict):
    job = get_current_job()
    job_id = job.id if job else None
    trace_id = payload.get("trace_id")
    start_ts = time.time()

    print(f"WORKER_RECEIVED trace_id={trace_id}", flush=True)

    try:
        chat_type = payload.get("chat_type", "unknown")
        to_id = payload.get("to_id")
        user_text = (payload.get("user_text") or "").strip()
        ts = payload.get("timestamp")

        if not to_id:
            raise ValueError("payload missing to_id")
        if not user_text:
            raise ValueError("payload missing user_text")

        append_to_daily_journal([f"- [{now_hm()}] ({chat_type}) Sonya: {user_text}"])

        reply_text, used_gemini = _build_reply_text(user_text)
        print(f"GEMINI_DONE trace_id={trace_id}", flush=True)

        # Use direct HTTP push with timeout to ensure timeouts are enforced
        try:
            resp = line_push_request(to_id, [reply_text], timeout=(3, 10))
            if not (200 <= getattr(resp, 'status_code', 0) < 300):
                raise RuntimeError(f"push failed status={getattr(resp,'status_code',None)} body={getattr(resp,'text',None)}")
            print(f"PUSH_DONE trace_id={trace_id}", flush=True)
        except Exception as e:
            print(f"PUSH_FAIL trace_id={trace_id} err={type(e).__name__}:{e}", flush=True)
            raise

        append_to_daily_journal([f"- [{now_hm()}] Bot: {reply_text}", ""])

        elapsed_ms = int((time.time() - start_ts) * 1000)
        print(
            f"[grand-healing] JOB_DONE trace_id={trace_id} elapsed_ms={elapsed_ms}",
            flush=True,
        )

    except Exception as e:
        print(f"JOB_FAIL trace_id={trace_id} err={type(e).__name__}:{e}", flush=True)
        # Re-raise so RQ can record failure / retry if configured
        raise


if __name__ == "__main__":
    redis_url = (
        os.environ.get("REDIS_URL")
        or os.environ.get("RAILWAY_REDIS_URL")
        or "redis://localhost:6379/0"
    )

    def _mask_redis(url: str | None) -> str:
        if not url:
            return ""
        try:
            return __import__("re").sub(r"://.*@", "://***@", url)
        except Exception:
            return url

    redis_conn = Redis.from_url(redis_url)

    queue = Queue(QUEUE_NAME, connection=redis_conn)

    print(f"WORKER_BOOT queue={QUEUE_NAME} redis={_mask_redis(redis_url)}", flush=True)

    worker = Worker([queue], connection=redis_conn)
    worker.work()
