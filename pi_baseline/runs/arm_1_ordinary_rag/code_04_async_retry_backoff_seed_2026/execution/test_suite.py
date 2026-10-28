from retry_caller import compute_delay

def test_exponential_backoff():
    assert compute_delay(1, base_delay=1.0) == 1.0
    assert compute_delay(2, base_delay=1.0) == 2.0
    assert compute_delay(3, base_delay=1.0) == 4.0
    assert compute_delay(10, base_delay=1.0, max_delay=32.0) == 32.0

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
