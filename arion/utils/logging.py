"""AutoDist logger."""

import logging as _logging
import os
import sys as _sys
import traceback as _traceback
import threading
import datetime

import autodist.const

_logger = None
_logger_lock = threading.Lock()

log_file_path = os.path.join(autodist.const.DEFAULT_WORKING_DIR,
                             datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log')
if not os.path.exists(autodist.const.DEFAULT_WORKING_DIR):
    os.makedirs(autodist.const.DEFAULT_WORKING_DIR)
default_log_format = '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s'


# Hao: Below two functions are copied from TensorFlow
def _get_caller(offset=3):
    """Returns a code and frame object for the lowest non-logging stack frame."""
    # Use sys._getframe().  This avoids creating a traceback object.
    # pylint: disable=protected-access
    f = _sys._getframe(offset)
    # pylint: enable=protected-access
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return code, f
        f = f.f_back
    return None, None


# The definition of `findCaller` changed in Python 3.2
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:
    def _logger_find_caller(stack_info=False): 
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():
        code, frame = _get_caller(4)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'


def get_logger():
    """
    Return AutoDist logger instance.

    # Simplify the logger code from TensorFlow, and combine it with the piper logger
    # tf_logging: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/platform/tf_logging.py
    # piper logger: https://gitlab.int.petuum.com/internal/piper-team/piper/blob/master/piper/core/utils/logger.py
    """
    global _logger
    if _logger:
        return _logger
    _logger_lock.acquire()

    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('autodist')
        logger.findCaller = _logger_find_caller
        # Note: If users use ABSL logging, the formatter and handlers will be rewritten following absl's 
        # as ABSL hacks into python logging and adds a handler to its root logger.
        if not _logging.getLogger().handlers:
            formatter = _logging.Formatter(default_log_format)
            # create file handler
            file_handler = _logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # by default _sys.stderr
            stream_handler = _logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()


def log(level, msg, *args, **kwargs):
    """Log a message at a given level."""
    get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log a message at the DEBUG level."""
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log a message at the ERROR level."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log a message at the CRITICAL level."""
    get_logger().critical(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log a message at the INFO level."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a message at the WARNING level."""
    get_logger().warning(msg, *args, **kwargs)


def set_verbosity(v):
    """Set the verbosity of autodist logger."""
    get_logger().setLevel(v)


def get_verbosity():
    """Get the verbosity of autodist logger."""
    return get_logger().getEffectiveLevel()
