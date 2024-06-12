import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os.path import join

from torch.utils.data import Dataset
from collections import defaultdict

import socket
from config import CommonConfig
from torch.utils.data import TensorDataset

from torchaudio.datasets import SPEECHCOMMANDS

import torch
import os

def get_dataset_path():
    hostname = socket.gethostname()
    if '407' in hostname:
        return '/data0/jmyan/dataset'
    else:
        return '/data/jmyan/dataset'

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target

class LabelwisePartitioner(object):
    def __init__(self, data, partition_sizes, seed, class_num=0, labels=None):
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        if hasattr(data, 'classes'):
            for class_idx in range(len(data.classes)):
                label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        elif hasattr(data, 'labels'):
            for class_idx in range(class_num): 
                label_indexes.append(list(np.where(np.array(data.labels) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        else:
            label_indexes = [list() for _ in range(class_num)]
            class_len = [0] * class_num
            # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
            for i, j in enumerate(data):
                if labels:
                    class_idx = labels.index(j[2])
                else:
                    class_idx = int(j[1])
                class_len[class_idx] += 1
                label_indexes[class_idx].append(i)
            # print(class_size)
            for i in range(class_num):
                rng.shuffle(label_indexes[i])
        
        # distribute class indexes to each vm according to sizes matrix
        try:
            for class_idx in range(len(data.classes)):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx
        except AttributeError:
            for class_idx in range(class_num):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]
        return selected_idxs
    
    def __len__(self):
        return len(self.data)



def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=False, num_workers=4, drop_last=True, collate_fn=None):
    if selected_idxs == None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, 
                                pin_memory=pin_memory, 
                                num_workers=num_workers,
                                collate_fn=collate_fn, 
                                drop_last=drop_last)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, 
                                batch_size=batch_size,
                                shuffle=shuffle, 
                                pin_memory=pin_memory, 
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                drop_last=drop_last)
    
    return DataLoaderHelper(dataloader)

def load_datasets(dataset_type):
    
    data_path = get_dataset_path()
    train_transform = load_default_transform(dataset_type, train=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)
    
    elif dataset_type == 'SPEECH':
        train_dataset = SubsetSC("training")
        test_dataset = SubsetSC("testing")
    
    elif dataset_type == 'UCIHAR':
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_","body_acc_y_","body_acc_z_",
            "body_gyro_x_","body_gyro_y_","body_gyro_z_",
            "total_acc_x_","total_acc_y_","total_acc_z_"
        ]

        # Output classes to learn how to classify
        LABELS = ["WALKING",
                  "WALKING_UPSTAIRS",
                  "WALKING_DOWNSTAIRS",
                  "SITTING",
                  "STANDING",
                  "LAYING"
                ]

        X_train_signals_paths = [data_path + "/UCIHAR/train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
        X_test_signals_paths = [data_path + "/UCIHAR/test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
        Y_train_path = data_path + "/UCIHAR/train/" + "y_train.txt"
        Y_test_path = data_path + "/UCIHAR/test/" + "y_test.txt"

        def load_X(X_signals_paths):
            X_signals = []
            for signal_type_path in X_signals_paths:
                file = open(signal_type_path, 'r')
                X_signals.append([np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in file]])
                file.close()
            return np.transpose(np.array(X_signals), (1, 2, 0))
        
        def load_Y(y_path):
            file = open(y_path, 'r')
            y_ = np.array([elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]],
                dtype=np.int32
            )
            file.close()
            return y_ - 1

        X_train = load_X(X_train_signals_paths)
        X_test = load_X(X_test_signals_paths)
        y_train = load_Y(Y_train_path)
        y_test = load_Y(Y_test_path)
        train_dataset = TensorDataset(torch.from_numpy(X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])), torch.from_numpy(y_train.reshape(-1)))
        test_dataset = TensorDataset(torch.from_numpy(X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])), torch.from_numpy(y_test.reshape(-1)))

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'IMAGE100':
        train_dataset = datasets.ImageFolder(join(data_path, 'IMAGE100/train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(join(data_path, "IMAGE100/test"), transform=test_transform)

    return train_dataset, test_dataset

def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'CIFAR100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])

    elif dataset_type == 'SVHN':
        dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    elif dataset_type == 'EMNIST':
        dataset_transform =  transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  
                        ])


    elif dataset_type == 'IMAGE100':
        dataset_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    else:
        return None

    return dataset_transform

def get_num_classes(dataset_type):
    if dataset_type == 'CIFAR10' or dataset_type == 'FashionMNIST' or dataset_type == 'MNIST':
        return 10
    elif dataset_type == 'EMNIST':
        return 62
    elif dataset_type == 'CIFAR100' or dataset_type == 'IMAGE100':
        return 100
    elif dataset_type == 'UCIHAR':
        return 6
    elif dataset_type == 'SPEECH':
        return 35
    else:
        return None

def partition_data(common_config: CommonConfig) -> LabelwisePartitioner:
    '''
    There are three commonly used non-iid settings, which id denoted by parameter 'non_iid_mode'
    non_iid_mode = 0: Each worker holds all classes of data, but one class accounts for a large proportion
    non_iid_mode = 1: Each worker is missing some classes, and the data size of every class is the same
    non_iid_mode = 2: Dirichlet distribution
    '''

    dataset_type = common_config.dataset_type
    non_iid_mode = common_config.non_iid_mode
    non_iid_ratio = common_config.non_iid_ratio
    # worker_num = common_config.worker_num
    worker_num = common_config.dataset_num


    train_dataset, _ = load_datasets(dataset_type)
    train_class_num = get_num_classes(dataset_type)

    if common_config.dataset_type == 'SPEECH':
        labels = load_speech_labels()
    else:
        labels = None

    
    if non_iid_mode == 0:
        if non_iid_ratio == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        else:
            ratio = non_iid_ratio * 0.1
            frac = ratio / ((1 - ratio) / (train_class_num - 1))
            big_num = int(worker_num / train_class_num)
            small_num = worker_num - big_num
            small_ratio = 1.0 / (big_num * frac + small_num)
            big_ratio = small_ratio * frac
            partition_sizes = np.ones((train_class_num, worker_num)) * (small_ratio)
            for i in range(worker_num):
                partition_sizes[i % train_class_num][i] = big_ratio
    
    elif non_iid_mode == 1:
        # 计算出每个 worker 缺少多少类数据
        missing_class_num = int(round(train_class_num * (non_iid_ratio * 0.1)))
        # 初始化分配矩阵
        partition_sizes = np.ones((train_class_num, worker_num))
        begin_idx = 0
        for worker_idx in range(worker_num):
            for i in range(missing_class_num):
                partition_sizes[(begin_idx + i) % train_class_num][worker_idx] = 0.
            begin_idx = (begin_idx + missing_class_num) % train_class_num
        for i in range(train_class_num):
            count = np.count_nonzero(partition_sizes[i])
            for j in range(worker_num):
                if partition_sizes[i][j] == 1.:
                    partition_sizes[i][j] = 1. / count
    
    elif non_iid_mode == 2:
        partition_sizes = []
        np.random.seed(common_config.numpy_seed)
        for _ in range(train_class_num):
            partition_sizes.append(np.random.dirichlet([non_iid_ratio] * worker_num))
        partition_sizes = np.array(partition_sizes)
    

    print('Data Partition:')
    for i in range(train_class_num):
        print('[{:2f} '.format(partition_sizes[i][0]), end=' ')
        for j in range(1, worker_num - 1):
            print(' {:2f} '.format(partition_sizes[i][j]), end=' ')
        print(' {:2f}]'.format(partition_sizes[i][worker_num - 1]))


    data_partition = LabelwisePartitioner(train_dataset, partition_sizes, common_config.random_seed,
                                          train_class_num, labels)
    return data_partition

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset=None, partition=None):
        super().__init__("/data/jmyan/dataset/speech/", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            if partition is not None:
                tmp_walker = [w for w in self._walker if w not in excludes]
                self._walker = [w for idx, w in enumerate(tmp_walker) if idx in partition]
            else:
                self._walker = [w for w in self._walker if w not in excludes]


import json
def load_speech_labels():
    labels_path = get_dataset_path() + '/speech/labels.json'
    with open(labels_path) as labels_json:
        labels = json.loads(labels_json.read())
    return labels

def collate_fn(batch, labels):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [torch.tensor(labels.index(label))]

    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def count_dataset(loader : DataLoaderHelper, dataset_type: str) -> np.ndarray:
    counts = np.zeros(get_num_classes(dataset_type))
    loader = loader.loader
    for _, target in loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    return counts

# def load_shakespeare():
#     train_dataset = ShakeSpeare(train=True)
#     test_dataset = ShakeSpeare(train=False)

#     return train_dataset, test_dataset

# class ShakeSpeare(Dataset):
#     def __init__(self, train=True):
#         super(ShakeSpeare, self).__init__()
#         train_clients, train_groups, train_data_temp, test_data_temp = read_data('/data0/jmyan/dataset/shakespeare/train', '/data0/jmyan/dataset/shakespeare/test')
#         self.train = train

#         if self.train:
#             self.dic_users = {}
#             train_data_x = []
#             train_data_y = []
#             for i in range(len(train_clients)):
#                 # if i == 100:
#                 #     break
#                 self.dic_users[i] = set()
#                 l = len(train_data_x)
#                 cur_x = train_data_temp[train_clients[i]]['x']
#                 cur_y = train_data_temp[train_clients[i]]['y']
#                 for j in range(len(cur_x)):
#                     self.dic_users[i].add(j + l)
#                     train_data_x.append(cur_x[j])
#                     train_data_y.append(cur_y[j])
#             self.data = train_data_x
#             self.label = train_data_y
#         else:
#             test_data_x = []
#             test_data_y = []
#             for i in range(len(train_clients)):
#                 cur_x = test_data_temp[train_clients[i]]['x']
#                 cur_y = test_data_temp[train_clients[i]]['y']
#                 for j in range(len(cur_x)):
#                     test_data_x.append(cur_x[j])
#                     test_data_y.append(cur_y[j])
#             self.data = test_data_x
#             self.label = test_data_y

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sentence, target = self.data[index], self.label[index]
#         indices = word_to_indices(sentence)
#         target = letter_to_vec(target)
#         # y = indices[1:].append(target)
#         # target = indices[1:].append(target)
#         indices = torch.LongTensor(np.array(indices))
#         # y = torch.Tensor(np.array(y))
#         # target = torch.LongTensor(np.array(target))
#         return indices, target

#     def get_client_dic(self):
#         if self.train:
#             return self.dic_users
#         else:
#             exit("The test dataset do not have dic_users!")


# def read_data(train_data_dir, test_data_dir):
#     '''parses data in given train and test data directories

#     assumes:
#     - the data in the input directories are .json files with
#         keys 'users' and 'user_data'
#     - the set of train set users is the same as the set of test set users

#     Return:
#         clients: list of client ids
#         groups: list of group ids; empty list if none found
#         train_data: dictionary of train data
#         test_data: dictionary of test data
#     '''
#     train_clients, train_groups, train_data = read_dir(train_data_dir)
#     test_clients, test_groups, test_data = read_dir(test_data_dir)

#     assert train_clients == test_clients
#     assert train_groups == test_groups

#     return train_clients, train_groups, train_data, test_data


# def read_dir(data_dir):
#     clients = []
#     groups = []
#     data = defaultdict(lambda: None)

#     files = os.listdir(data_dir)
#     files = [f for f in files if f.endswith('.json')]
#     for f in files:
#         file_path = os.path.join(data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         data.update(cdata['user_data'])

#     clients = list(sorted(data.keys()))
#     return clients, groups, data


# # ------------------------
# # utils for shakespeare dataset

# ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
# NUM_LETTERS = len(ALL_LETTERS)
# # print(NUM_LETTERS)

# def _one_hot(index, size):
#     '''returns one-hot vector with given size and value 1 at given index
#     '''
#     vec = [0 for _ in range(size)]
#     vec[int(index)] = 1
#     return vec


# def letter_to_vec(letter):
#     '''returns one-hot representation of given letter
#     '''
#     index = ALL_LETTERS.find(letter)
#     return index


# def word_to_indices(word):
#     '''returns a list of character indices

#     Args:
#         word: string
    
#     Return:
#         indices: int list with length len(word)
#     '''
#     indices = []
#     for c in word:
#         indices.append(ALL_LETTERS.find(c))
#     return indices