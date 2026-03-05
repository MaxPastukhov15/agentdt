import logging.config
from pythonjsonlogger import jsonlogger

LOGGING = {
    'version' : 1,
    'disable_existing_loggers' : False,
    'formatters': {
        "json": {
            "format": "%(asctime)s %(levelname)s %(message)s %(module)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }

    },
    "handlers" : {
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/app.log",  # Файл создастся в корне проекта
            "formatter": "json",
        }
    },
    "loggers": {"" : {"handlers" : ["file"], "level" : "DEBUG"}},

}

logging.config.dictConfig(LOGGING)
