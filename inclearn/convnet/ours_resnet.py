"""Taken & slightly modified from:
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MultiBN(nn.Module):

    def __init__(self, inplanes):
        super(MultiBN, self).__init__()
        self.task_num = 1
        self.inplanes = inplanes
        self.BN_list = nn.ModuleList()
        self.BN_list.append(nn.BatchNorm2d(inplanes))

    def add_bn(self):
        new_bn = nn.BatchNorm2d(self.inplanes)
        nn.init.constant_(new_bn.weight, 1)
        nn.init.constant_(new_bn.bias, 0)
        new_bn.train()
        self.BN_list.append(new_bn)

    def get_bn_num(self):
        return len(self.BN_list)

    def forward(self, x, task_id):
        out = self.BN_list[task_id](x)
        return out


class Ordinary_Sequential(nn.Module):

    def __init__(self, *args):
        super(Ordinary_Sequential, self).__init__()
        self.module_list = nn.ModuleList()
        for module in args:
            self.module_list.append(module)

    def forward(self, x, task_id):

        for module in self.module_list:
            if isinstance(module, MultiBN):
                x = module(x, task_id)
            else:
                x = module(x)

        return x

class Block_Sequential(nn.Module):

    def __init__(self, *args):
        super(Block_Sequential, self).__init__()
        self.module_list = nn.ModuleList()
        for module in args:
            self.module_list.append(module)

    def forward(self, x, task_id):
        for module in self.module_list:
            x = module(x, task_id)

        return x


class BasicBlock_expandable(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock_expandable, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MultiBN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MultiBN(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x, task_id):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, task_id)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, task_id)

        if self.downsample is not None:
            identity = self.downsample(x, task_id)

        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)
        return out



class ResNet_exp1(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 nf=64,
                 zero_init_residual=True,
                 dataset='cifar',
                 start_class=0,
                 remove_last_relu=False):
        super(ResNet_exp1, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        if 'cifar' in dataset:
            self.conv1 = Ordinary_Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                       MultiBN(nf), nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            if start_class == 0:
                self.conv1 =  Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                # Following PODNET implmentation
                self.conv1 = Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(BasicBlock_expandable, 1 * nf, layers[0])
        self.layer2 = self._make_layer(BasicBlock_expandable, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer_ordinary(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer_ordinary(block, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Ordinary_Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                MultiBN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return Block_Sequential(*layers)

    def _make_layer_ordinary(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def add_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.add_bn()

    def enable_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.BN_list[-1].train()
                m.BN_list[-1].weight.requires_grad = True
                m.BN_list[-1].bias.requires_grad = True

    def freeze_old_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                for i in range(len(m.BN_list)-1):
                    m.BN_list[i].eval()
                    m.BN_list[i].weight.requires_grad = False
                    m.BN_list[i].bias.requires_grad = False

    def forward(self, x, task_id):
        x = self.conv1(x, task_id)
        x = self.layer1(x, task_id)
        x = self.layer2(x, task_id)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class ResNet_exp2(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 nf=64,
                 zero_init_residual=True,
                 dataset='cifar',
                 start_class=0,
                 remove_last_relu=False):
        super(ResNet_exp2, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        if 'cifar' in dataset:
            self.conv1 = Ordinary_Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                       MultiBN(nf), nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            if start_class == 0:
                self.conv1 =  Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                # Following PODNET implmentation
                self.conv1 = Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(BasicBlock_expandable, 1 * nf, layers[0])
        self.layer2 = self._make_layer(BasicBlock_expandable, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock_expandable, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer4(block, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Ordinary_Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                MultiBN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return Block_Sequential(*layers)

    def _make_layer4(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def add_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.add_bn()

    def enable_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.BN_list[-1].train()
                m.BN_list[-1].weight.requires_grad = True
                m.BN_list[-1].bias.requires_grad = True

    def freeze_old_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                for i in range(len(m.BN_list)-1):
                    m.BN_list[i].eval()
                    m.BN_list[i].weight.requires_grad = False
                    m.BN_list[i].bias.requires_grad = False

    def forward(self, x, task_id):
        x = self.conv1(x, task_id)
        x = self.layer1(x, task_id)
        x = self.layer2(x, task_id)
        x = self.layer3(x, task_id)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet_exp3(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 nf=64,
                 zero_init_residual=True,
                 dataset='cifar',
                 start_class=0,
                 remove_last_relu=False):
        super(ResNet_exp3, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        if 'cifar' in dataset:
            self.conv1 = Ordinary_Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                       MultiBN(nf), nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            if start_class == 0:
                self.conv1 =  Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                # Following PODNET implmentation
                self.conv1 = Ordinary_Sequential(
                    nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                    MultiBN(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(BasicBlock_expandable, 1 * nf, layers[0])
        self.layer2 = self._make_layer(BasicBlock_expandable, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock_expandable, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock_expandable, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Ordinary_Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                MultiBN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return Block_Sequential(*layers)

    def _make_layer4(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def add_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.add_bn()

    def enable_new_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                m.BN_list[-1].train()
                m.BN_list[-1].weight.requires_grad = True
                m.BN_list[-1].bias.requires_grad = True

    def freeze_old_task_bn(self):
        for m in self.modules():
            if isinstance(m, MultiBN):
                for i in range(len(m.BN_list)-1):
                    m.BN_list[i].eval()
                    m.BN_list[i].weight.requires_grad = False
                    m.BN_list[i].bias.requires_grad = False

    def forward(self, x, task_id):
        x = self.conv1(x, task_id)
        x = self.layer1(x, task_id)
        x = self.layer2(x, task_id)
        x = self.layer3(x, task_id)
        x = self.layer4(x, task_id)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 nf=64,
                 zero_init_residual=True,
                 dataset='cifar',
                 start_class=0,
                 remove_last_relu=False):
        super(ResNet, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        if 'cifar' in dataset:
            self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            if start_class == 0:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                # Following PODNET implmentation
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(nf),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(block, 1 * nf, layers[0])
        self.layer2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet18_exp1(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet_exp1(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet18_exp2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet_exp2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_exp3(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet_exp3(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model