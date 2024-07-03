import logging

def set_logging_level(level_str):
    level_dict = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET
    }
    logging_level = level_dict.get(level_str.upper(), logging.INFO)
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

