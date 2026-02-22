from avala._pagination import CursorPage


def test_cursor_page_has_more():
    page = CursorPage(items=[1, 2, 3], next_cursor="abc")
    assert page.has_more is True
    assert len(page) == 3


def test_cursor_page_no_more():
    page = CursorPage(items=[1, 2], next_cursor=None)
    assert page.has_more is False


def test_cursor_page_iteration():
    page = CursorPage(items=["a", "b", "c"])
    assert list(page) == ["a", "b", "c"]
