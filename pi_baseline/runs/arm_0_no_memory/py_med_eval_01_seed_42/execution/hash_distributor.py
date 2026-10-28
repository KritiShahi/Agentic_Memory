import asyncio
import hashlib

def _hash_worker(data):
    if not isinstance(data, (bytes, str)):
        raise TypeError("Invalid payload type")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

async def compute_payload_hashes(payloads):
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _hash_worker, p) for p in payloads]
    return list(await asyncio.gather(*tasks))
