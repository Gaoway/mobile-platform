import os
import time
import argparse
import asyncio
import copy
import numpy as np
import random
import torch
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train
import datasets, models
from mpi4py import MPI
from utils import *
import setproctitle


epoch = 1
comm = MPI.COMM_WORLD
client_config = ClientConfig()

async def get_config():
    global comm, epoch, client_config
    config_received= await get_data(comm, MASTER_RANK, epoch)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

parser = argparse.ArgumentParser(description='Worker')
parser.add_argument('--config_file', type=str, default='default')
args = parser.parse_args()


rank = comm.Get_rank()
csize = comm.Get_size()
MASTER_RANK = 0


scheme = os.getcwd().split('/')[-1]
config_file = args.config_file

common_config : CommonConfig = load_common_config(scheme, config_file)
random.seed(common_config.random_seed)
np.random.seed(common_config.numpy_seed)
torch.manual_seed(common_config.torch_seed)
torch.cuda.manual_seed_all(common_config.torch_seed)
setproctitle.setproctitle('worker-{}'.format(rank))
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % get_gpu_num())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_dataset, _ = datasets.load_datasets(common_config.dataset_type)
if common_config.dataset_type == 'SPEECH':
    labels = datasets.load_speech_labels()
local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)


def main():

    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
                asyncio.ensure_future(
                    get_config()
                )
            )
        tasks.append(
            asyncio.ensure_future(
                local_training()
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        if epoch == common_config.epoch + 1:
            break


async def local_training():

    global comm, epoch, common_config, client_config, local_model, train_dataset, labels, device
    load_start_time = time.time()
    if common_config.dataset_type == 'SPEECH':
        train_loader = datasets.create_dataloaders(train_dataset, 
                                                   client_config.batch_size,
                                                   client_config.train_data_idxes,
                                                   collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        train_loader = datasets.create_dataloaders(train_dataset, 
                                                   client_config.batch_size,
                                                   client_config.train_data_idxes)
    # 更新本地参数

    global_param = copy.deepcopy(client_config.global_param)

    # 加载本地模型
    torch.nn.utils.vector_to_parameters(global_param, local_model.parameters())

    local_model.load_state_dict(client_config.bn_param, strict=False)

    client_config.load_time = time.time() - load_start_time

    epoch_lr = max(common_config.lr * (common_config.decay_rate ** epoch), common_config.lr_min)

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=epoch_lr,
        momentum=common_config.momentum,
        weight_decay=common_config.weight_decay)
    
    train_start_time = time.time()
    client_config.train_loss, client_config.train_correct, client_config.train_samples = train(
        local_model, 
        train_loader,
        optimizer,
        device,
        client_config.tau
    )

    client_config.train_time = time.time() - train_start_time
    
    # 取出本地参数
    client_config.local_param = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    
    # 计算本地梯度
    client_config.local_grad = client_config.local_param - client_config.global_param


    client_config.bn_param = models.get_bn_dict(local_model)

    client_config.param_num, client_config.size = models.get_param_size(local_model)


    # 发送本轮状态
    await send_data(comm, client_config, MASTER_RANK, epoch)

    epoch += 1

    train_loader = None


    

if __name__ == '__main__':
    main()
