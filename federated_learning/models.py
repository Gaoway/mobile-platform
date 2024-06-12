import math
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.nn as nn

from typing import Tuple

pers_prefix = {
    'mobilenetv2': [
        'layers.13',
        'layers.14',
        'layers.15',
        'layers.16',
        'conv2',
        'bn2',
        'linear'
    ]
}


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device):
        super(RNN, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn_ih_l0 = nn.Linear(self.embed_size, self.hidden_size)
        self.rnn_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.hidden_size, device=self.device).requires_grad_()
        embeds = self.embedding(x)
        for i in range(len(embeds[0])):
            h = self.rnn_hh_l0(h) + self.rnn_ih_l0(embeds[:, i])
            h = torch.tanh(h)
        out_put = self.fc1(h)
        return out_put
    
def create_rnn_instance(model_ratio, device):
    if True:
        embed_size = 256
        hidden_size = int(256 * model_ratio)
        return RNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=80, device=device)
    else:
        raise ValueError("Not valid model type")


def split_param(model_or_param, model_type : str) -> Tuple[dict, dict]:

    if isinstance(model_or_param, nn.Module):
        params = model_or_param.state_dict()
    elif isinstance(model_or_param, dict):
        params = model_or_param

    global_param = dict()
    person_param = dict()
    for name, value in params.items():
        if name not in pers_prefix[model_type]:
            global_param[name] = value
        else:
            person_param[name] = value

    return global_param, person_param


def get_param_size(model_or_param) -> Tuple[int, float]:

    if isinstance(model_or_param, nn.Module):
        params = model_or_param.state_dict()
    elif isinstance(model_or_param, dict):
        params = model_or_param

    num_params = 0
    param_size = 0.0
    for _, param in params.items():
        param_size += param.nelement() * param.element_size()
        num_params += param.nelement()
    return num_params, param_size / 1024 / 1024

def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

class HeteroCNN(nn.Module):
    def __init__(self, ratio, data_shape=[3, 32, 32], classes_size=10):
        super(HeteroCNN, self).__init__()
        # torch.manual_seed(cfg['model_init_seed'])

        hidden_size = [
            int(64 * ratio),
            int(128 * ratio),
            int(128 * ratio)
        ]

        # head
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1)]

        # hidden layers
        for i in range(len(hidden_size) - 1):
            blocks.extend([
                nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        # classifier
        blocks.extend([
            nn.Flatten(),
            nn.Linear(
                int(hidden_size[-1] * 32 * 32 / (4 ** (len(hidden_size)-1))),
                classes_size
            )
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.blocks(x)
        return output

def create_cnn(model_ratio):
    model = HeteroCNN(model_ratio)
    model.apply(init_param)
    return model    


def create_model_instance(dataset_type, model_type):
    if dataset_type == 'CIFAR10':
        if model_type == 'mobilenetv2':
            model = MobileNetV2_CIFAR(num_classes=10)
        elif model_type == 'densenet':
            model = densenet_cifar(num_classes=10)
        elif model_type == 'resent':
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        elif model_type == 'alexnet':
            model = AlexNet(class_num=10)
    
    if dataset_type == 'CIFAR100':
        if model_type == 'mobilenetv2':
            model = MobileNetV2_CIFAR(num_classes=100)
        elif model_type == 'densenet':
            model = densenet_cifar(num_classes=100)
        elif model_type == 'resnet':
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
    
    if dataset_type == 'IMAGE100':
        if model_type == 'mobilenetv2':
            model = MobileNetV2_IMAGE100()
        elif model_type == 'resnet':
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
    
    if dataset_type == 'UCIHAR':
        if model_type == 'cnn':
            model = CNN_HAR()
    
    if dataset_type == 'SPEECH':
        if model_type == 'cnn':
            model = CNN_S()
        
    return model


'''
MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_CIFAR(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_CIFAR, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MobileNetV2_IMAGE100(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=100):
        super(MobileNetV2_IMAGE100, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

# def densenet_cifar10():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

# def densenet_cifar100():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=100)

def densenet_cifar(num_classes : int):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=num_classes)


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

class AlexNet(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class CNN_HAR(nn.Module):
    def __init__(self):
        super(CNN_HAR, self).__init__()
        
        width = 1
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=int(12*width),          #输出高度12
                      kernel_size=3,            #卷积核尺寸3*3
                      stride=1,
                      padding=1),               #(1*128*9)-->(12*128*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #(12*128*9)-->(12*64*4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(12*width),
                      out_channels=int(32*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),               #(12*64*4)-->(32*64*4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # (32*32*2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(32*width),
                      out_channels=int(64*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),                #(32*32*2)-->(64*32*2)
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(64*32*2*width), int(1024*width)), 
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(int(1024*width),6)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1) 
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class CNN_S(nn.Module):
    def __init__(self, n_input=1, stride=16, n_channel=32, n_output=35):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.fc_layer = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=2).squeeze()
        


import copy

def get_bn_dict(model: nn.Module) -> dict:
    bn_dict = {}
    for key, value in model.state_dict().items():
        for name in ['running_mean', 'running_var', 'num_batches_tracked']:
            if name in key:
                bn_dict[key] = copy.deepcopy(value.detach())
    return bn_dict

