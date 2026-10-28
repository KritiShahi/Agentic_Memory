from version_parser import parse_semver

def test_basic_semver():
    assert parse_semver('1.2.3') == {'major': 1, 'minor': 2, 'patch': 3, 'tag': None}

def test_prerelease_semver():
    res = parse_semver('2.0.1-rc.1')
    assert res == {'major': 2, 'minor': 0, 'patch': 1, 'tag': 'rc.1'}

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
