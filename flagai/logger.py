# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import logging
import sys
import os
import torch.distributed as dist

is_bmt = 0
try:
    import bmtrain as bmt
    is_bmt = 1
except:
    is_bmt = 0

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


if 'logger' not in dir():
    logger = LoggerFactory.create_logger(name="FlagAI", level=logging.INFO)
while len(logger.handlers) > 1:
    # Why is this happening?
    logger.removeHandler(logger.handlers[-1])


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when one of following condition meets
    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]
    Args:
        message (str)
        ranks (list)
        level (int)
    """
    
    my_rank = -1
    if is_bmt and bmt.init.is_initialized():
        should_log = not bmt.init.is_initialized()
        my_rank = bmt.rank() if bmt.init.is_initialized() else -1
    else:
        should_log = not dist.is_initialized()
        my_rank = dist.get_rank() if dist.is_initialized() else -1
        
    ranks = ranks or []
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        final_message = "[Rank {}] {}".format(my_rank, message)
        logger.log(level, final_message)

def print_json_dist(message, ranks=None, path=None):
    """Print message when one of following condition meets
    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]
    Args:
        message (str)
        ranks (list)
        path (str)
    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        message['rank'] = my_rank
        import json
        with open(path, 'w') as outfile:
            json.dump(message, outfile)
            os.fsync(outfile)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string
    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.
    Example:
        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level_str = max_log_level_str.lower()
    if max_log_level_str not in log_levels:
        raise ValueError(
            f"{max_log_level_str} is not one of the `logging` levels")

    return get_current_level() <= log_levels[max_log_level_str]
