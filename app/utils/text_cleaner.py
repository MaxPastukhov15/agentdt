import re


def clean_text(text: str, mode: str = "user") -> str:
    """
    Universal text cleaner.

    Modes:
    - 'user': Strict query cleaning.
    - 'content': Cleaning data from HTML/PDF.
    """

    if not text:
        return ""

    match mode:
        case "user":
            pattern = r"[^А-Яа-яA-Za-z0-9!?.,;:()\ '\-_]"
            cleaned = re.sub(pattern, "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        case "content":
            pattern = r"[^А-Яа-яA-Za-z0-9!?.,;:()\ \"'\+\-\_%°/=\[\]\*\^\n\u2013\u2014]"
            cleaned = re.sub(pattern, "", text)

            cleaned = re.sub(r"[ \t]+", " ", cleaned)

            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

            cleaned = "\n".join([line.strip() for line in cleaned.split("\n")])

        case "data":
            text = re.sub(r"_(.*?)_", r"\1", text)
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

            text = re.sub(r"\[([+\-−eE])\]", r"\1", text)
            text = re.sub(r"\[([A-Za-zА-Яа-я0-9\s/.,]{1,15})\]", r"\1", text)

            text = re.sub(r"\[([=\-;:|])\]", r"\1", text)

            text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

            text = re.sub(r"[\uFFFD]", " ", text)

            text = re.sub(r"\s+([.,;:!?)\]])", r"\1", text)
            text = re.sub(r"([(\[])\s+", r"\1", text)

            text = re.sub(r"[ \t]+", " ", text)
            cleaned = re.sub(r"\n{3,}", "\n\n", text)
        case _:
            return text.strip()

    return cleaned.strip()
