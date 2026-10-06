import sqlite3


def find_user(connection: sqlite3.Connection, user_id: str):
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()
