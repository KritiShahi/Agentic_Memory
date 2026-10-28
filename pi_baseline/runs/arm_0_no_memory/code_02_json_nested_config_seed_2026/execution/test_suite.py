from config_loader import get_nested_key

def test_nested_found():
    cfg = {'db': {'port': 5432}}
    assert get_nested_key(cfg, 'db.port') == 5432

def test_nested_primitive_intermediate():
    cfg = {'db': 'localhost'}
    assert get_nested_key(cfg, 'db.port.sub', default_val=8000) == 8000

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
