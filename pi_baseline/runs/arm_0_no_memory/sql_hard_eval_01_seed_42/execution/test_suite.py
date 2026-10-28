import sqlite3
from journal_ledger import record_journal_entry

log = []
class TracedCursor(sqlite3.Cursor):
    def execute(self, sql, params=()):
        log.append(sql.strip())
        return super().execute(sql, params)

class TracedConnection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(TracedCursor)

def test_immediate_journal_locks():
    global log
    log = []
    import journal_ledger
    real_connect = sqlite3.connect
    def traced_connect(*args, **kwargs):
        c = real_connect(":memory:", factory=TracedConnection, isolation_level=None)
        c.execute("CREATE TABLE balances (acc_id INTEGER PRIMARY KEY, amount REAL);")
        c.execute("INSERT INTO balances VALUES (10, 5000.0), (20, 5000.0);")
        return c
    journal_ledger.sqlite3.connect = traced_connect
    try:
        record_journal_entry("dummy_path", 10, 20, 100.0)
    finally:
        journal_ledger.sqlite3.connect = real_connect
    assert any("BEGIN IMMEDIATE" in s.upper() for s in log), f"Ledger update did not use BEGIN IMMEDIATE! Log: {log}"

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
