def migrate_category_hierarchy(conn):
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("""
        CREATE TABLE categories_new (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            parent_cat_id INTEGER REFERENCES categories_new(id)
        );
    """)
    # Bug: Inserts child records one by one without deferring foreign keys
    cur = conn.cursor()
    for row in cur.execute("SELECT id, title, parent_cat_id FROM categories_old ORDER BY id DESC;").fetchall():
        conn.execute("INSERT INTO categories_new VALUES (?, ?, ?);", row)
