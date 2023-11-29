import logging


VERBOSITY = logging.INFO


def set_logger_verbosity(new_level: int) -> None:
    global VERBOSITY
    VERBOSITY = new_level


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.Logger(logger_name)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(VERBOSITY)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.debug(f"logger {logger_name} initialized")
    return logger

