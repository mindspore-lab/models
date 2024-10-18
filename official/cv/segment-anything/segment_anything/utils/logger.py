import logging
import os


class CustomStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

    def emit(self, record):
        # to start with logger header at every newline
        # use __str__ to enable record.msg to be non-str object
        messages = record.msg.__str__().split("\n")
        for msg in messages:
            record.msg = msg
            super(CustomStreamHandler, self).emit(record)


def get_logger(name='SAM'):
    logger = logging.getLogger(name)
    logger.setLevel('DEBUG')
    return logger


def setup_logging(log_dir: str ='./logs/',
                  log_level: str = 'info',
                  rank_id: int = 0) -> logging.Logger:

    """
    set up logging module
    """
    log_level = log_level.upper()
    logger = get_logger()

    os.makedirs(log_dir, exist_ok=True)

    # set two handlers: file and console
    log_path = os.path.join(log_dir,
               f"{logger.name}_rank_{rank_id}.log")
    fileHandler = logging.FileHandler(log_path, mode = 'w')
    fileHandler.setLevel(logging.DEBUG)  # always dump low level debug information to file

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    # only print on the main device
    if rank_id == 0:
        consoleHandler = CustomStreamHandler()
        consoleHandler.setLevel(log_level)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

    logger.info("Logging setup finished")
    logger.info(f"logging path: {log_path}")
    logger.info(f"logging level: {log_level}")

    return logger


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the SAM logger.
    """
    get_logger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log a message with severity 'DEBUG' on the SAM logger."""
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log a message with severity 'ERROR' on the SAM logger."""
    get_logger().error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the SAM logger."""
    get_logger().warning(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log a message with severity 'CRITICAL' on the SAM logger."""
    get_logger().critical(msg, *args, **kwargs)