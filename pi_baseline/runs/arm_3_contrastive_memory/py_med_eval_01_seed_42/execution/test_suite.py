import asyncio
from hash_distributor import compute_payload_hashes

def test_hash_computation():
    async def run():
        hashes = await compute_payload_hashes(["alpha", "beta"])
        assert len(hashes) == 2
        assert isinstance(hashes[0], str) and len(hashes[0]) == 64
    asyncio.run(run())

def test_exception_propagation():
    async def run_err():
        try:
            await compute_payload_hashes([12345])
            assert False, "Should have raised TypeError"
        except TypeError:
            pass
    asyncio.run(run_err())

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
