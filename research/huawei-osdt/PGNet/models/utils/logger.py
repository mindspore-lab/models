import logging
import os
import sys
from typing import Optional

__all__ = [
    "set_logger",
]

logger_initialized = {}


def set_logger(
    name: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_fn: Optional[str] = None,
    rank: int = 0,
    log_level: Optional[str] = logging.INFO,
) -> logging.Logger:
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    fmt = "%(asctime)s %(name)s %(levelname)s - %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if log_fn is None:
            log_fn = "log_%s.txt" % rank
        file_handler = logging.FileHandler(os.path.join(output_dir, log_fn))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger
