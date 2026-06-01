import json
import os
from config.config import settings
def load_chat_memory(thread_id: str) -> str:
    """Подгружает сохраненное summary для конкретного чата"""
    file_path = settings.long_term_memory / f"{thread_id}_summary.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("summary", "")
    return ""

def save_chat_memory(thread_id: str, summary: str):
    """Сохраняет обновленное summary на диск"""
    with open(settings.long_term_memory / f"{thread_id}_summary.json", 'w', encoding='utf-8') as f:
        json.dump({"summary": summary}, f, ensure_ascii=False)