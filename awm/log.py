import logging
import sys
from typing import TextIO

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Modules started with `python -m` (e.g. awm.core.server) log as "__main__".
AWM_LOGGER_NAMES = ("awm", "__main__")


def setup_logging(level: int = logging.DEBUG, stream: TextIO = sys.stderr) -> None:
    """Send awm's log records to `stream`, replacing any earlier setup.

    Only awm's own loggers get the handler, so libraries keep their default
    (warnings and above) instead of flooding the output with debug records.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    for name in AWM_LOGGER_NAMES:
        logger = logging.getLogger(name)
        for old in list(logger.handlers):
            logger.removeHandler(old)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
