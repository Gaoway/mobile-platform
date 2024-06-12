import os
import argparse
import asyncio
import random
import numpy as np
import torch
import copy
from config import *
import datasets, models
from mpi4py import MPI
from utils import *
import setproctitle
from training_utils import test



parser = argparse.ArgumentParser(description='Parameter Server')
parser.add_argument('--config_file', type=str, default='default')
args = parser.parse_args()
setproctitle.setproctitle('server')

# 编号排名最后的 GPU 最不容易被分配到，故将 server 放在最后一个 GPU 上
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = str(get_gpu_num() - 1)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()


scheme = os.getcwd().split('/')[-1]
config_file = args.config_file
selected_worker_num = int(csize) - 1

# 初始化 logger, 加载 common_config
common_config = load_common_config(scheme, config_file)
logger = create_logger(scheme, config_file, common_config)

assert selected_worker_num == common_config.selected_worker_num



random.seed(common_config.random_seed)
np.random.seed(common_config.numpy_seed)
torch.manual_seed(common_config.torch_seed)
torch.cuda.manual_seed_all(common_config.torch_seed)



def main():
    logger.info(f'scheme: {scheme}')
    logger.info(f'csize:{int(csize)}')
    logger.info(f'server start (rank):{int(rank)}')
    for key, value in common_config.__dict__.items():
        logging.info(f'{key}: {value}')


    worker_config_list = [ClientConfig() for _ in range(common_config.worker_num)]

    # 这里创建 activated worker 的列表
    worker_list: List[Worker] = list()
    for worker_idx in range(common_config.selected_worker_num):
        worker_list.append(Worker(config=ClientConfig(), 
                                  rank=worker_idx+1))



    _, test_dataset = datasets.load_datasets(common_config.dataset_type)
    if common_config.dataset_type == 'SPEECH':
        labels = datasets.load_speech_labels()
        test_loader = datasets.create_dataloaders(test_dataset, common_config.test_batch_size, shuffle=False,
                                                  collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        test_loader = datasets.create_dataloaders(test_dataset, common_config.test_batch_size, shuffle=False)
    
    partitioner = datasets.partition_data(common_config)

    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    

    # 输出每个虚拟 worker 的数据分配情况

    for worker_idx, worker_config in enumerate(worker_config_list):
        worker_config.train_data_idxes = partitioner.use(worker_idx % common_config.dataset_num)
        worker_config.sample_size = len(worker_config.train_data_idxes)
        worker_config.avg_weight = 1 / selected_worker_num
        worker_config.idx = worker_idx

    # 全局的 Metrics
    total_upload_traffic: float = 0.
    total_download_traffic: float = 0.
    best_test_acc: float = 0.
    best_epoch: int = 0

    for epoch_idx in range(1, 1 + common_config.epoch):

        logger.info(f'\n**********************Epoch[{epoch_idx}/{common_config.epoch}]**********************')

        # 激活客户端
        selected_worker_config_list = random.sample(worker_config_list, common_config.selected_worker_num)

        logger.info('Activate Workers:')

        global_param = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()

        for idx, worker in enumerate(worker_list):
            worker.config = copy.deepcopy(selected_worker_config_list[idx])
            worker.config.global_param = global_param
            worker.config.bn_param = models.get_bn_dict(global_model)
            worker.config.batch_size = common_config.batch_size
            worker.config.tau = common_config.tau
            worker.config.idx = idx




        
        # 激活 worker, 分发全局模型
        communication_parallel(worker_list, epoch_idx, comm, action='send_config')

        # 收集被激活 worker 的本地模型, 并聚合成全局模型
        communication_parallel(worker_list, epoch_idx, comm, action="get_config")

       

        local_train_acc_list = list()
        local_train_loss_list = list()

        
        # 输出本轮的训练结果
        for worker in worker_list:
            
            # 将 worker 的状态保存下来
            worker_config_list[worker.config.idx] = copy.deepcopy(worker.config)

            logger.info(f'worker-{worker.config.idx}:')
            logger.info(f'Samples size is {worker.config.sample_size}.')


            # 记录训练性能
            train_acc = 100 * worker.config.train_correct / worker.config.train_samples

            logger.info(f'Train Loss {worker.config.train_loss:.4f}; '
                        f'Accuracy {train_acc:.2f}% [{worker.config.train_correct}/{worker.config.train_samples}]')
        
            local_train_acc_list.append(train_acc)
            local_train_loss_list.append(worker.config.train_loss)


            download_traffic, download_param_num = models.get_param_size(global_model)
            # 记录训练开销
            logger.info(f'Load Time: {worker.config.load_time:.4f}s; ' 
                        f'Training Time: {worker.config.train_time:.4f}s; '
                        f'Upload Traffic {worker.config.size:.4f} MB {worker.config.param_num}.'
                        f'Download Traffic {download_traffic:.4f} MB {download_param_num}.')
            total_upload_traffic += worker.config.size
            total_download_traffic += download_traffic


        
        local_avg_train_acc = sum(local_train_acc_list) / len(local_train_acc_list)
        local_avg_train_loss = sum(local_train_loss_list) / len(local_train_loss_list)
        logger.info(f'\nLocal Avg Train Accuracy {local_avg_train_acc:.2f}% Loss {local_avg_train_loss:.4f}\n')

        logging.info('################# GLOBAL AGGREGATION #################')

        # 计算聚合权重

        aggregate_param(global_model, worker_list)
        test_start_time = time.time()
        test_loss, test_correct, test_samples = test(global_model, test_loader, device)
        test_time = time.time() - test_start_time
        logging.info(f'Test Finish, Time Cost is {test_time:4f} s')
        test_acc = 100 * test_correct / test_samples
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch_idx
        logger.info(f'Global Test Loss {test_loss:.4f}; '
                    f'Accuracy {test_acc:.2f}% [{test_correct}/{test_samples}]; '
                    f'Best Test Accuracy is {best_test_acc}% in epoch {best_epoch}.')
        logger.info(f'Total Upload Traffic is {total_upload_traffic:4f} MB; '
                    f'Total Download Traffic is {total_download_traffic:4f} MB; '
                    f'Total Traffic is {total_download_traffic + total_upload_traffic:4f} MB.')



def aggregate_param(global_model: torch.nn.Module, 
                    worker_list: List[Worker]) -> dict:
    # 计算聚合权重
    total_samples_num = 0
    for worker in worker_list:
        total_samples_num += worker.config.sample_size
    for worker in worker_list:
        worker.config.avg_weight = worker.config.sample_size / total_samples_num
        logging.info(f'The average weight of worker {worker.config.idx} is {worker.config.avg_weight:.4f}.')
    
    # 聚合本地模型
    with torch.no_grad():
        # 聚合参数
        global_param = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
        global_grad = torch.zeros_like(global_param)
        for worker in worker_list:
            global_grad += worker.config.avg_weight * worker.config.local_grad
        global_param += global_grad
        torch.nn.utils.vector_to_parameters(global_param, global_model.parameters())

        # 聚合 BN 层 ！！！！非常重要 ！！！！
        global_bn = models.get_bn_dict(global_model)
        for key, value in global_bn.items():
            global_bn[key] = torch.zeros_like(value)
            for worker in worker_list:
                if 'num_batches_tracked' in key:
                    global_bn[key] += worker.config.bn_param[key]
                else:
                    global_bn[key] += worker.config.bn_param[key] * worker.config.avg_weight
        global_model.load_state_dict(global_bn, strict=False)


def communication_parallel(worker_list: List[Worker], 
                           epoch_idx: int, 
                           comm: MPI.COMM_WORLD, 
                           action: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for worker in worker_list:
        if action == 'send_config':
            task = asyncio.ensure_future(worker.send_config(comm, epoch_idx))
        elif action == 'get_config':
            task = asyncio.ensure_future(worker.get_config(comm, epoch_idx))
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == "__main__":
    main()
