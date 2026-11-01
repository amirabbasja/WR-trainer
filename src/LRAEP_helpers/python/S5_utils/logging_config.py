import os
import sys
from loguru import logger

def setup_logging(level: str='INFO') -> None:
    logger.remove()
    format_string = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <level>{level}</level> - <white>{message}</white> - <blue>{file}:{line}</blue>'
    log_level = os.environ.get('LRAEP_LOG_LEVEL', 'INFO')
    logger.add(sys.stderr, level=log_level.upper(), format=format_string, colorize=True, backtrace=True, diagnose=True, enqueue=True)

def get_logger(name: str):
    return logger.bind(name=name)
should_log = os.environ.get('LRAEP_LOG_LEVEL', 'INFO').upper() != 'NONE'
if should_log:
    setup_logging()
