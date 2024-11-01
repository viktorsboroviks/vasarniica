"""
Logging.
"""

import logging
import sys
import typing


LOGGER_NAME = "vlog"
LOG_PREFIX = "[v] "
LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_LEVEL_DEFAULT = "info"

# pylint: disable=invalid-name
# this is a global variable
# and its name is intended to be written in lower case
g_logger = None


def configure(
    log_level: typing.Literal[
        "debug", "info", "warning", "error", "critical"
    ] = LOG_LEVEL_DEFAULT
):
    """
    Configure logging.
    If not called, default configuration would be used.
    """
    assert log_level in LOG_LEVEL

    # log handler is used as a channel for outputting logs
    # it has its own configureation, including formatting and level
    # there may be several handlers attached to one g_logger
    lh = logging.StreamHandler(sys.stdout)
    lh.setFormatter(logging.Formatter(LOG_PREFIX + "%(msg)s"))

    # pylint: disable=global-statement
    # this variable is intended to be global to keep the logger status
    global g_logger
    g_logger = logging.getLogger(LOGGER_NAME)
    g_logger.setLevel(LOG_LEVEL[log_level])
    g_logger.addHandler(lh)


def is_effective(
    log_level: typing.Literal["debug", "info", "warning", "error", "critical"]
) -> bool:
    """
    Return True if provided log_level is effective.
    """
    assert log_level in LOG_LEVEL

    if g_logger is None:
        configure()
    # log level is an integres, starting with 'debug'
    # ref: https://docs.python.org/3/howto/logging.html#logging-levels
    return g_logger.getEffectiveLevel() <= LOG_LEVEL[log_level]


def debug(msg):
    """
    Log msg with severity 'debug'.
    """
    if g_logger is None:
        configure()
    g_logger.debug(msg)


def info(msg):
    """
    Log msg with severity 'info'.
    """
    if g_logger is None:
        configure()
    g_logger.info(msg)


def warning(msg):
    """
    Log msg with severity 'warning'.
    """
    if g_logger is None:
        configure()
    g_logger.warning(msg)


def error(msg):
    """
    Log msg with severity 'error'.
    """
    if g_logger is None:
        configure()
    g_logger.error(msg)


def critical(msg):
    """
    Log msg with severity 'critical'.
    """
    if g_logger is None:
        configure()
    g_logger.critical(msg)
