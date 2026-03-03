import logging
import os
from datetime import datetime

_logger = None

def get_logger():
    global _logger
    if _logger is not None:
        return _logger

    # Create logs directory in the project root (football_analytics_framework)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with current datetime
    log_filename = datetime.now().strftime("%Y%m%d_%H_%M.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure the logger
    _logger = logging.getLogger("football_analytics")
    _logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger already exists
    if not _logger.handlers:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        _logger.addHandler(file_handler)
        _logger.addHandler(console_handler)
        
    return _logger
