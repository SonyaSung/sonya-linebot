import os
import sys
import redis
from rq import Worker, Queue


def _version_fingerprint():
    """Return a compact fingerprint to prove which code is running."""
    commit = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT") or "unknown"
    service = os.getenv("RAILWAY_SERVICE_NAME") or os.getenv("RAILWAY_PROJECT_NAME") or "unknown"
    return f"commit={commit} service={service} file={__file__}"


def process_line_job(payload: dict):
    """
    RQ job entrypoint. app.py enqueues "worker.process_line_job".

    Requirements:
    - Import-safe: do NOT read REDIS_URL or connect to Redis at import time.
    - Minimal behavior: push a reply back to LINE user if configured.
    - Logs: include PUSH_ATTEMPT/PUSH_OK/PUSH_FAIL and a VERSION fingerprint.
    - TODO: wire this to the real slow task handler later.
    """
    import time

    # ---- Always print version fingerprint so Railway logs prove code is updated
    print(f"[worker] VERSION ts={int(time.time())} {_version_fingerprint()}", flush=True)

    # ---- Validate payload
    if not isinstance(payload, dict):
        print(f"[worker] BAD_PAYLOAD type={type(payload).__name__}", flush=True)
        return {"ok": False, "reason": "bad_payload_type", "payload_type": type(payload).__name__}

    trace_id = payload.get("trace_id")
    chat_type = payload.get("chat_type")
    to_id = payload.get("to_id")
    user_text = (payload.get("user_text") or "").strip()
    payload_keys = list(payload.keys())

    # ---- Minimal stub return (kept even if push fails)
    result = {
        "ok": True,
        "trace_id": trace_id,
        "payload_keys": payload_keys,
    }

    # ---- Only push back for 1:1 user chat (group/room typically needs reply token; push to group/room is different)
    if chat_type != "user":
        print(f"[worker] SKIP_PUSH trace_id={trace_id} reason=chat_type_not_user chat_type={chat_type}", flush=True)
        result["push"] = {"ok": False, "reason": "chat_type_not_user"}
        return result

    if not to_id:
        print(f"[worker] SKIP_PUSH trace_id={trace_id} reason=missing_to_id", flush=True)
        result["push"] = {"ok": False, "reason": "missing_to_id"}
        return result

    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print(f"[worker] SKIP_PUSH trace_id={trace_id} reason=missing_LINE_CHANNEL_ACCESS_TOKEN", flush=True)
        result["push"] = {"ok": False, "reason": "missing_token"}
        return result

    # ---- Construct minimal message
    reply_text = f"trace_id={trace_id}\n你說：{user_text}" if trace_id else f"你說：{user_text}"

    print(f"[worker] PUSH_ATTEMPT trace_id={trace_id} to_id={to_id} text_len={len(reply_text)}", flush=True)

    try:
        # line-bot-sdk v3
        from linebot.v3.messaging import (
            ApiClient,
            Configuration,
            MessagingApi,
            PushMessageRequest,
            TextMessage,
        )

        configuration = Configuration(access_token=token)
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)
            req = PushMessageRequest(
                to=to_id,
                messages=[TextMessage(text=reply_text)],
            )
            resp = messaging_api.push_message(req)

        print(f"[worker] PUSH_OK trace_id={trace_id} to_id={to_id} resp={type(resp).__name__}", flush=True)
        result["push"] = {"ok": True}
        return result

    except Exception as e:
        print(
            f"[worker] PUSH_FAIL trace_id={trace_id} to_id={to_id} "
            f"err={type(e).__name__}:{e}",
            flush=True,
        )
        result["push"] = {"ok": False, "reason": "exception", "err_type": type(e).__name__}
        return result


def main():
    # Only the worker runtime requires Redis; keep import safe.
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("ERROR: Missing REDIS_URL (set it in Railway Variables or your shell env).", file=sys.stderr)
        raise SystemExit(1)

    queue_name = os.getenv("RQ_QUEUE", "line")
    conn = redis.from_url(redis_url)
    queue = Queue(queue_name, connection=conn)

    print(f"[worker] START queue={queue_name} redis_url_set={'yes' if bool(redis_url) else 'no'} {_version_fingerprint()}", flush=True)

    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()