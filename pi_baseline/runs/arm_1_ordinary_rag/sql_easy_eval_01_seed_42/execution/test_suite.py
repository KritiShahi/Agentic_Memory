import sqlite3
from product_view_mgr import migrate_products_and_view

def test_view_accessibility():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
    conn.execute("INSERT INTO products VALUES (1, 'Widget', 19.99);")
    conn.execute("CREATE VIEW active_products_view AS SELECT id, name, price FROM products;")
    migrate_products_and_view(conn)
    row = conn.execute("SELECT id, name, retail_price FROM active_products_view WHERE id = 1;").fetchone()
    assert row == (1, 'Widget', 19.99)

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
