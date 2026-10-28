def migrate_products_and_view(conn):
    conn.execute("DROP VIEW IF EXISTS active_products_view;")
    conn.execute("DROP TABLE IF EXISTS products_old;")
    conn.execute("ALTER TABLE products RENAME TO products_old;")
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, retail_price REAL, is_available INTEGER);")
    conn.execute("INSERT INTO products SELECT id, name, price, 1 FROM products_old;")
    conn.execute("CREATE VIEW active_products_view AS SELECT id, name, retail_price FROM products WHERE is_available = 1;")
    conn.execute("DROP TABLE products_old;")
