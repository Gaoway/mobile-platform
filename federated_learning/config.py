from comm_utils import *
from typing import List

class CommonConfig:
    def __init__(self):
        # 模型与数据类型
        self.dataset_type : str = 'CIFAR10'
        self.model_type : str = 'mobilenetv2'
        # 数据分配 
        self.non_iid_mode : int = 2
        self.non_iid_ratio : int = 0.1
        self.train_portion : float = 0.75

        # 一般模型训练相关超参
        self.batch_size : int = 32
        self.test_batch_size : int = 128
        self.lr : float = 0.1
        self.lr_min : float = 0.005
        self.decay_rate : float = 0.993
        self.weight_decay : float = 3e-4
        self.momentum : float = 0.9
        # 训练次数
        self.epoch : int = 500
        self.tau : int = 35
        # 各种种子
        self.random_seed : int = 2024
        self.torch_seed : int = 2024
        self.numpy_seed : int = 2024

        # 客户端数量
        self.worker_num: int = 100
        self.selected_worker_num : int = 10
        self.dataset_num : int = 100






class ClientConfig:
    def __init__(self):
        
        self.idx = None

        # local dataset
        self.train_data_idxes = None
        self.sample_size = None
        self.class_count = None
        self.class_var = None

        # control variable
        self.avg_weight = None

        self.batch_size = None
        self.tau = None

        # model parameters
        self.global_param = None
        self.local_param = None
        self.local_grad = None

        self.bn_param : dict = None


        # communication overhead
        self.size = None
        self.param_num = None

        # computation overhead
        self.train_time = None
        self.load_time = None

        # training performance
        self.train_loss = None
        self.train_correct = None
        self.train_samples = None






class Worker:
    def __init__(self, config: ClientConfig, rank: int):
        #这个 config 就是后面的 client_config
        self.config = config
        self.rank = rank

    async def get_config(self, comm, epoch_idx):
        self.config = await get_data(comm, self.rank, epoch_idx)
    
    async def send_config(self, comm, epoch_idx):
        await send_data(comm, self.config, self.rank, epoch_idx)
