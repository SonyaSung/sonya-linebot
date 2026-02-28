import os
import sys
import redis
from rq import Worker, Queue

def process_line_job(payload: dict):
    """
    RQ job entrypoint. app.py enqueues 'worker.process_line_job'.
    Import-safe: do NOT require REDIS_URL or connect to Redis at import time.
    TODO: wire this to the real slow job handler.
    """
    if isinstance(payload, dict):
        trace_id = payload.get("trace_id")
        keys = list(payload.keys())
    else:
        trace_id = None
        keys = []
    return {"ok": True, "trace_id": trace_id, "payload_keys": keys}

def main():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("ERROR: Missing REDIS_URL (set it in Railway Variables or your shell env).", file=sys.stderr)
        raise SystemExit(1)

    queue_name = os.getenv("RQ_QUEUE", "line")
    conn = redis.from_url(redis_url)
    queue = Queue(queue_name, connection=conn)

    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
