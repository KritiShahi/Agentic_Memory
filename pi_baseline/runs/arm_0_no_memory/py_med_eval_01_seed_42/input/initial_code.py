import asyncio
import hashlib

def _hash_worker(data):
    if not isinstance(data, (bytes, str)):
        raise TypeError("Invalid payload type")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

async def compute_payload_hashes(payloads):
    loop = asyncio.get_event_loop()
    # Bug: creating futures without awaiting gather or propagating worker errors properly
    futures = [loop.run_in_executor(None, _hash_worker, p) for p in payloads]
    return futures
