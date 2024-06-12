from numpy import intersect1d
from datasets import *
from config import *
from training_utils import *
from models import *

common_config = CommonConfig()

# common_config.non_iid_mode = 0
# common_config.non_iid_ratio = 0
# common_config.dataset_num = 50

BS = 2048
train_frac = 0.8
EPOCH = 50

train_dataset, test_dataset = load_datasets('CIFAR10')
# data_partition = partition_data(common_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_idxes = data_partition.use(0)
# random.seed(2024)
# random.shuffle(data_idxes)
# train_data_volume = int(len(data_idxes) * train_frac)
# train_idxes = data_idxes[:train_data_volume]
# test_idxes = data_idxes[train_data_volume:]

train_loader = create_dataloaders(train_dataset, 
                                  batch_size=BS, 
                                #   selected_idxs=train_idxes
                                  )

test_loader = create_dataloaders(test_dataset,
                                 batch_size=256,
                                 shuffle=False
                                 )

model = create_model_instance('CIFAR10', 'alexnet')
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=common_config.lr,
    momentum=common_config.momentum,
    weight_decay=common_config.weight_decay)

for epoch in range(EPOCH):
    train_loss, train_correct, train_samples = train(model, 
                                                     train_loader, 
                                                     optimizer, 
                                                     device)
    
    train_acc = 100 * train_correct / train_samples
    # test_loss, test_correct, test_samples = test(
    #     model,
    #     test_loader, 
    #     device
    # )
    print(f'Iteration [{epoch}/{EPOCH}] Training Loss {train_loss}, {train_acc}% [{train_correct}/{train_samples}]')
    # test_acc = 100 * test_correct / test_samples
    # print(f'Test Loss {test_loss}, {test_acc}% [{test_correct}/{test_samples}]')

    test_loss, test_correct, test_samples = test(
            model,
            test_loader, 
            device
    )
    test_acc = 100 * test_correct / test_samples
    print(f'Test Loss {test_loss}, {test_acc}% [{test_correct}/{test_samples}]')
        
    
