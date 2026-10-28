import sqlite3

def record_journal_entry(db_path, debit_account, credit_account, amount):
    conn = sqlite3.connect(db_path, timeout=10.0)
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    cur.execute("UPDATE balances SET amount = amount - ? WHERE acc_id = ?", (amount, debit_account))
    cur.execute("UPDATE balances SET amount = amount + ? WHERE acc_id = ?", (amount, credit_account))
    conn.commit()
    conn.close()
