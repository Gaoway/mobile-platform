import os
import time
import logging
from config import CommonConfig
import socket
import json

def get_gpu_num() -> int:
    hostname = socket.gethostname()
    if '402' in hostname:
        return 2
    elif '405' in hostname:
        return 8
    else:
        return 4

def get_root_path() -> str:
    hostname = socket.gethostname()
    if '407' in hostname:
        return '/data0/jmyan'
    else:
        return '/data/jmyan'

def create_logger(scheme : str, config_name: str, common_config: CommonConfig):
    now = time.strftime("%Y-%m-%d_%H:%M", time.localtime(time.time()))
    RESULT_PATH = get_root_path() + f'/record/{scheme}'
    os.makedirs(RESULT_PATH, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(
        filename=os.path.join(RESULT_PATH, f'{common_config.dataset_type}-{config_name}_at_{now}.log')
    )
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger

def load_common_config(scheme : str, config_file :str):
    config_path = get_root_path() + f'/{scheme}/configs/{config_file}.json'
    with open(config_path) as json_file:
        content = json.loads(json_file.read())
    common_config = CommonConfig()
    for k, v in content.items():
        setattr(common_config, k, v)
    return common_config
            


