import logging
from rich.logging import RichHandler
from pathlib import Path

def get_logger(name: str = "csv_samurai", level: str = "INFO") -> logging.Logger:
    FORMAT = "[%(asctime)s] | %(message)s"
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(LOG_DIR / "app.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FORMAT,datefmt="[%X]"))

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter(FORMAT,datefmt="[%X]"))

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = get_logger()