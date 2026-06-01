from app.utils.text_cleaner import clean_text


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_user_mode_removes_special_chars():
    text = "Hello! Привет... (test) спец_символы: @#$%^&*"
    result = clean_text(text, mode="user")
    assert "Hello" in result
    assert "Привет" in result
    assert "@" not in result
    assert "#" not in result
    assert "$" not in result


def test_clean_text_user_mode_collapses_spaces():
    text = "Hello    World\n\n\nTest"
    result = clean_text(text, mode="user")
    assert "  " not in result
    assert "\n" not in result


def test_clean_text_user_mode_keeps_basic_punctuation():
    text = "Что такое pH? Это: 7.0!; (кислота)"
    result = clean_text(text, mode="user")
    assert "?" in result
    assert "." in result
    assert ";" in result
    assert "(" in result
    assert ")" in result


def test_clean_text_content_mode_keeps_newlines():
    text = "Line1\n\nLine2\n\n\nLine3"
    result = clean_text(text, mode="content")
    assert "Line1" in result
    assert "Line2" in result
    assert "Line3" in result


def test_clean_text_content_mode_normalizes_excessive_newlines():
    text = "A\n\n\n\n\nB"
    result = clean_text(text, mode="content")
    assert result == "A\n\nB"


def test_clean_text_data_mode_removes_markdown_brackets():
    text = "_italic_ and **bold** text"
    result = clean_text(text, mode="data")
    assert "_italic_" not in result
    assert "italic" in result


def test_clean_text_data_mode_removes_artifact_brackets():
    text = "[+] [e] [-] test"
    result = clean_text(text, mode="data")
    assert "[+]" not in result
    assert "[e]" not in result


def test_clean_text_data_mode_removes_standalone_numbers():
    text = "text\n42\nmore"
    result = clean_text(text, mode="data")
    assert "42" not in result.splitlines()


def test_clean_text_data_mode_fixes_spacing():
    text = "text . more , list ;"
    result = clean_text(text, mode="data")
    assert "text ." not in result
    assert "text." in result


def test_clean_text_unknown_mode_returns_stripped():
    text = "  hello world  "
    result = clean_text(text, mode="invalid")
    assert result == "hello world"
