import logging
from rich.logging import RichHandler

def get_logger(name: str = "csv_samurai", level: str = "INFO") -> logging.Logger:
    FORMAT = "[%(asctime)s] | [%(levelname)s] | %(message)s"
    logging.basicConfig(
        level = level.upper(),
        format = FORMAT,
        datefmt = "[%X]",
        handlers = [RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(name)