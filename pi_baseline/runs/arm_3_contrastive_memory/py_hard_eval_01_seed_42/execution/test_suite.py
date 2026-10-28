import asyncio
from bounded_channel import BoundedAsyncChannel

def test_close_channel_wakes_waiters():
    async def run():
        chan = BoundedAsyncChannel(capacity=1)
        await chan.put("first")

        # Second put will wait
        put_task = asyncio.create_task(chan.put("second"))
        await asyncio.sleep(0.01)

        # Closing channel must wake up put_task with exception rather than deadlocking
        await chan.close()
        try:
            await asyncio.wait_for(put_task, timeout=0.2)
            assert False, "Should have raised RuntimeError"
        except (RuntimeError, asyncio.CancelledError):
            pass
    asyncio.run(run())

if __name__ == '__main__':
    import sys
    failed = False
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except Exception as e:
                print(f"FAIL: {name}: {e}", file=sys.stderr)
                failed = True
    if failed:
        sys.exit(1)
