import logging

DEFAULT_LOGGER_NAME = 'yolov5rtsp'
DEFAULT_LOGGER_FORMATTER = '%(asctime)s %(name)s %(levelname)s %(message)s'

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def get_logger(name=DEFAULT_LOGGER_NAME, level=INFO, formatter=DEFAULT_LOGGER_FORMATTER):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(DEFAULT_LOGGER_FORMATTER))
    logger.addHandler(ch)

    return logger
