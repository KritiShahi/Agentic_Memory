import sqlite3
from cat_migration import migrate_category_hierarchy

def test_migration_out_of_order():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("CREATE TABLE categories_old (id INTEGER, title TEXT, parent_cat_id INTEGER);")
    conn.execute("INSERT INTO categories_old VALUES (2, 'Laptops', 1), (1, 'Computers', NULL);")
    migrate_category_hierarchy(conn)
    category_total = conn.execute("SELECT COUNT(*) FROM categories_new;").fetchone()[0]
    assert category_total == 2, f"Expected 2 migrated categories, got {category_total}"

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
