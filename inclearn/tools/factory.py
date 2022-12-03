import torch
from torch import nn
from torch import optim

from inclearn import models
from inclearn.convnet import resnet, cifar_resnet, modified_resnet_cifar, preact_resnet, ours_resnet
from inclearn.datasets import data


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet32":
        return cifar_resnet.resnet32()
    elif convnet_type == "modified_resnet32":
        return modified_resnet_cifar.resnet32(**kwargs)
    elif convnet_type == "preact_resnet18":
        return preact_resnet.PreActResNet18(**kwargs)
    elif convnet_type == "resnet18_exp1":
        return ours_resnet.resnet18_exp1(**kwargs)
    elif convnet_type == "resnet18_exp2":
        return ours_resnet.resnet18_exp2(**kwargs)
    elif convnet_type == "resnet18_exp3":
        return ours_resnet.resnet18_exp3(**kwargs)
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset):
    if cfg["model"] == "incmodel":
        return models.IncModel(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "weight_align":
        return models.Weight_Align(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "bic":
        return models.BiC(cfg, trial_i, _run, ex, tensorboard, inc_dataset)


    if cfg["model"] == "ensmodel0":
        return models.EnsModel0(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel1":
        return models.EnsModel1(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "sepmodel0":
        return models.SepModel0(cfg, trial_i, _run, ex, tensorboard, inc_dataset)

    if cfg["model"] == "ensmodel2":
        return models.EnsModel2(cfg, trial_i, _run, ex, tensorboard, inc_dataset)

    if cfg["model"] == "ensmodel3":
        return models.EnsModel3(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel4":
        return models.EnsModel4(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel5":
        return models.EnsModel5(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel6":
        return models.EnsModel6(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel7":
        return models.EnsModel7(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "ensmodel8":
        return models.EnsModel8(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "expnet1":
        return models.ExpNet1(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "expnet2":
        return models.ExpNet2(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "expnet3":
        return models.ExpNet3(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    else:
        raise NotImplementedError(cfg["model"])


def get_data(cfg, trial_i):
    return data.IncrementalDataset(
        trial_i=trial_i,
        dataset_name=cfg["dataset"],
        random_order=cfg["random_classes"],
        shuffle=True,
        batch_size=cfg["batch_size"],
        workers=cfg["workers"],
        validation_split=cfg["validation"],
        resampling=cfg["resampling"],
        increment=cfg["increment"],
        data_folder=cfg["data_folder"],
        start_class=cfg["start_class"],
    )


def set_device(cfg):
    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    cfg["device"] = device
    return device
