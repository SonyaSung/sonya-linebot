import os

from redis import Redis
from rq import Queue, Worker

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
    messaging_api,
    now_hm,
)


QUEUE_NAME = os.environ.get("RQ_QUEUE", "default")


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
            generation_config={"temperature": 0.2, "max_output_tokens": 80},
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
            generation_config={"temperature": 0.4, "max_output_tokens": 120},
        )
        lines = _first_n_lines(text or "", 3)
        if len(lines) == 3:
            return "\n".join(lines), True
        return AI_BUSY_TRANSLATION_MESSAGE, True

    return gemini_reply(prompt), True


def process_line_job(payload: dict):
    print("WORKER RECEIVED JOB:", payload, flush=True)
    print("WORKER GOT JOB", payload.keys(), flush=True)
    chat_type = payload.get("chat_type", "unknown")
    to_id = payload.get("to_id")
    user_text = (payload.get("user_text") or "").strip()
    ts = payload.get("timestamp")

    if not to_id:
        raise ValueError("payload missing to_id")
    if not user_text:
        raise ValueError("payload missing user_text")

    print(
        f"job start chat_type={chat_type} to_id={to_id} msg_len={len(user_text)} ts={ts}",
        flush=True
    )

    try:
        append_to_daily_journal([f"- [{now_hm()}] ({chat_type}) Sonya: {user_text}"])
        print("journal write user: success")
    except Exception as e:
        print(f"journal write user: failed {type(e).__name__}: {e}")
        raise

    try:
        reply_text, used_gemini = _build_reply_text(user_text)
        print(f"gemini step: {'success' if used_gemini else 'skipped'}")
    except Exception as e:
        print(f"gemini step: failed {type(e).__name__}: {e}")
        raise

    try:
        messaging_api.push_message(
            PushMessageRequest(
                to=to_id,
                messages=[TextMessage(text=reply_text)],
            )
        )
        print(f"push success to={to_id} reply_len={len(reply_text)}", flush=True)
    except Exception as e:
        print(f"push failed to={to_id}: {type(e).__name__}: {e}", flush=True)
        raise

    try:
        append_to_daily_journal([f"- [{now_hm()}] Bot: {reply_text}", ""])
        print("journal write bot: success")
    except Exception as e:
        print(f"journal write bot: failed {type(e).__name__}: {e}")
        raise

    print(f"job end chat_type={chat_type} to_id={to_id}", flush=True)


if __name__ == "__main__":
    redis_url = (
        os.environ.get("REDIS_URL")
        or os.environ.get("RAILWAY_REDIS_URL")
        or "redis://localhost:6379/0"
    )

    redis_conn = Redis.from_url(redis_url)

    queue = Queue(QUEUE_NAME, connection=redis_conn)

    print(f"Worker starting: queue={QUEUE_NAME} redis={redis_url}", flush=True)

    worker = Worker([queue], connection=redis_conn)
    worker.work()
