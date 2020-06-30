# Copyright 2020 Petuum. All Rights Reserved.
#
# It includes the derived work based on:
# https://github.com/tensorflow/tensorflow
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AutoDist logger."""

import datetime
import logging as _logging
import os
import sys as _sys
import threading
import traceback as _traceback

import autodist.const

_logger = None
_logger_lock = threading.Lock()

log_dir = os.path.join(autodist.const.DEFAULT_WORKING_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir,
                             datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log')
default_log_format = '[PID#%(process)s:%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s'


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
    """Get the AutoDist logger instance."""
    global _logger
    if _logger:
        return _logger
    _logger_lock.acquire()

    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('autodist')
        # If this attribute evaluates to true,
        # events logged to this logger will be passed to the handlers of higher level (ancestor) loggers,
        # in addition to any handlers attached to this logger.
        logger.propagate = False
        logger.findCaller = _logger_find_caller
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
