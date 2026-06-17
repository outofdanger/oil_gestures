import logging

from oil_gestures.core.constants import DEFAULT_LOGGER_NAME


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """
    Return a configured project logger.
    Use this function instead of raw print() calls in project modules.

    Example:
        from oil_gestures.core.logger import get_logger

        logger = get_logger(__name__)

        logger.info("MediaPipe landmarker initialized")
        logger.warning("No hand detected")
        logger.error("Camera cannot be opened")
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger