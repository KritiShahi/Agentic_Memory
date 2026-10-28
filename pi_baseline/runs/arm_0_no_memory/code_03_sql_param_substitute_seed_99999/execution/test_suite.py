from query_builder import build_query

def test_int_param():
    q = build_query('SELECT * FROM users WHERE age > :min_age', {'min_age': 21})
    assert q == 'SELECT * FROM users WHERE age > 21'

def test_str_param():
    q = build_query('SELECT * FROM users WHERE role = :role', {'role': 'admin'})
    assert q == 'SELECT * FROM users WHERE role = "admin"'

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
