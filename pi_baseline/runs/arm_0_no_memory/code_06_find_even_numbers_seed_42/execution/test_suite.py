from even_filter import find_even_numbers

def test_mixed_numbers():
    assert find_even_numbers([1, 2, 3, 4, 5, 6]) == [2, 4, 6]

def test_all_odd():
    assert find_even_numbers([1, 3, 5, 7]) == []

def test_negatives_and_zero():
    assert find_even_numbers([0, -2, -3, 8]) == [0, -2, 8]

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
