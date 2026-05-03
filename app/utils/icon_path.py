import os
import sys


def get_resource(path):
    # Если запущен .exe, файлы лежат в sys._MEIPASS
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    # Если мы в .exe, папка assets лежит в корне, а не на уровень выше
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(base, path)
    # Если запуск через python app/main.py, идем на уровень выше
    return os.path.join(base, "..", path)
