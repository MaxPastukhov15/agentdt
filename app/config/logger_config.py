import logging.config
import pythonjsonlogger
from .config import paths

log_dir = paths.data_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": pythonjsonlogger.json.JsonFormatter,
            "format": "%(asctime)s %(levelname)s %(message)s %(module)s",
        },
        "print": {"format": "%(levelname)s %(message)s", "class": "logging.Formatter"},
    },
    "handlers": {
        "file": {"class": "logging.FileHandler", "filename": "logs/app.log", "formatter": "json", "encoding": "utf-8"},
        "tools_file": {"class": "logging.FileHandler", "filename": "logs/tools.log", "formatter": "json", "encoding": "utf-8"},
        "db_file": {"class": "logging.FileHandler", "filename": "logs/db.log", "formatter": "json", "encoding": "utf-8"},
        "graph_stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "print",
        },
    },
    "loggers": {
        "": {"handlers": ["file"], "level": "INFO"},
        "tools": {
            "handlers": ["tools_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "db": {
            "handlers": ["db_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "graph": {
            "handlers": ["graph_stdout"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

LOGGING["handlers"]["file"]["filename"] = str(log_dir / "app.log")
LOGGING["handlers"]["tools_file"]["filename"] = str(log_dir / "tools.log")
LOGGING["handlers"]["db_file"]["filename"] = str(log_dir / "db.log")

logging.config.dictConfig(LOGGING)
