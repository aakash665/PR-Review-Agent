import sqlite3


def find_order(connection: sqlite3.Connection, order_id: str):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
    return cursor.fetchone()
