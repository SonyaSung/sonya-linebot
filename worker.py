import os
import redis
from rq import Worker, Queue

redis_url = os.getenv('REDIS_URL')
if not redis_url:
    raise RuntimeError("Missing REDIS_URL")

queue_name = os.getenv('RQ_QUEUE', 'line')
conn = redis.from_url(redis_url)
queue = Queue(queue_name, connection=conn)

if __name__ == '__main__':
    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False)
