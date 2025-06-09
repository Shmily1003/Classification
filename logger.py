import logging
import os
from datetime import datetime

try:
    import torch.distributed as dist
except ImportError:
    dist = None  # 非分布式环境兼容处理

def is_main_process():
    """判断是否为主进程（用于多进程日志保护）"""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def setup_my_logger(stream: dict, file: dict, cfg: dict) -> logging.Logger:
    """Setup a logger with both stream and file handlers.
    Args:
        stream (dict): _Description of the stream handler settings.
            stream['enable'] (bool): Whether to enable the stream handler.
            stream['level'] (int): Logging level for the stream handler.
        file (dict): _Description of the file handler settings.
            file['enable'] (bool): Whether to enable the file handler.
            file['level'] (int): Logging level for the file handler.
        cfg (dict): Configuration dictionary.
            cfg['log_root'] (str): Directory where log files will be saved.

    Returns:
        class:`logging.Logger`: Configured logger instance.
    """
    assert stream['enable'] or file['enable'], "At least one of stream or file must be True"
    logger = logging.getLogger('my_debug_logger')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            '**** {%(asctime)s} [%(filename)s] (Line:%(lineno)d) - %(levelname)s : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        if is_main_process():
            if file['enable']:
                os.makedirs(cfg['log_root'], exist_ok=True)
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                log_path = os.path.join(cfg['log_root'], f'{timestamp}.log')
                fh = logging.FileHandler(log_path, mode='w')
                fh.setLevel(file['level'])
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            if stream['enable']:
                ch = logging.StreamHandler()
                ch.setLevel(stream['level'])
                ch.setFormatter(formatter)
                logger.addHandler(ch)
    return logger