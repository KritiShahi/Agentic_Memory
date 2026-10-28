from binary_search import binary_search

def test_found_mid():
    assert binary_search([1, 3, 5, 7, 9, 11], 7) == 3

def test_found_bounds():
    assert binary_search([1, 3, 5, 7, 9, 11], 1) == 0
    assert binary_search([1, 3, 5, 7, 9, 11], 11) == 5

def test_not_found():
    assert binary_search([1, 3, 5, 7, 9, 11], 4) == -1
    assert binary_search([], 5) == -1

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
