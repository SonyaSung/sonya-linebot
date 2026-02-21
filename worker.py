import os
import redis
from rq import Worker, Queue
from rq.connections import Connection  # ✅ rq 2.x 在這裡

print("WORKER_BOOT queue=line redis=SET", flush=True)

redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise RuntimeError("Missing REDIS_URL")

conn = redis.from_url(redis_url)

queue = Queue("line", connection=conn)

print("WORKER_BOOT queue=line redis=OK", flush=True)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker([queue])
        worker.work(with_scheduler=False)
import os
import redis
from rq import Worker, Queue, Connection

print("WORKER_BOOT queue=line redis=SET", flush=True)

redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise RuntimeError("Missing REDIS_URL")

conn = redis.from_url(redis_url)

queue = Queue("line", connection=conn)

print("WORKER_BOOT queue=line redis=OK", flush=True)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker([queue])
        worker.work(with_scheduler=False)
