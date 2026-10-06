from app.service import find_user


def test_find_user_returns_row(database):
    assert find_user(database, "1") is not None
