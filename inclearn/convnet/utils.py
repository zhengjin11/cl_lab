import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter


def finetune_last_layer(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()
    optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss = criterion(outputs / temperature, targets)
            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
    return network


def finetune_last_layer_ens1(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()

    param_list = []
    for i in range(len(network.module.classifier)):
        param_list.extend(network.module.classifier[i].parameters())

    # optim = SGD(network.module.classifier[-1].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optim = SGD(param_list, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss = criterion(outputs / temperature, targets)
            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
    return network


def get_aux_loss(logits, targets, increments):
    aux_loss = torch.tensor(0.0).to(logits.device)

    count = 0
    base = 0
    for i in range(len(increments)):
        other_task_example_index = (targets<base) | (targets>=(base+increments[i]))
        if (torch.sum(other_task_example_index) != 0):
            count+=1
            other_task_example_logits = logits[other_task_example_index][:, base:base+increments[i]]
            aux_loss +=  -(other_task_example_logits.mean(1) - torch.logsumexp(other_task_example_logits,dim=1)).mean()
        base+=increments[i]

    aux_loss = (aux_loss/count)

    return aux_loss


def deep_finetune_last_layer_ens1(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    _increments=None,
    use_aux = True,
    aux_loss_weight=0.01,
    test_loader=None,
):

    network.eval()
    param_list = []

    for i in range(len(network.module.convnets)):
        m_num = 0
        for m in network.module.convnets[i].layer4.modules():
            if isinstance(m, nn.Conv2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            if isinstance(m, nn.BatchNorm2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)

                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)


    for i in range(len(network.module.convnets)):
        m_num = 0
        for m in network.module.convnets[i].layer3.modules():
            if isinstance(m, nn.Conv2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            if isinstance(m, nn.BatchNorm2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)

                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)


    for i in range(len(network.module.classifier)):
        param_list.extend(network.module.classifier[i].parameters())

    # optim = SGD(network.module.classifier[-1].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optim = SGD(param_list, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for e in range(nepoch):
        total_loss = 0.0
        total_aux_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for i, (inputs, targets) in enumerate(loader, start=1):
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss = criterion(outputs / temperature, targets)
            aux_loss = get_aux_loss(outputs / temperature, targets, _increments)

            total_loss += loss
            total_aux_loss += aux_loss

            if use_aux:
                loss+=aux_loss*aux_loss_weight
            loss.backward()
            optim.step()

            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (e, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning ce loss %.3f aux loss %.3f acc %.3f" %
                        (e, total_loss.item()/i, total_aux_loss.item()/i, total_correct.item() / total_count))
    return network



def deep_finetune_last_layer_ens7(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    _increments=None,
    use_aux = True,
    aux_loss_weight=0.01,
    test_loader=None,
):

    network.eval()
    param_list = []

    m_num = 0
    for m in network.module.convnet.layer4.modules():
        if isinstance(m, nn.Conv2d):
            m_num += 1
            m.train()
            for param in m.parameters():
                param.requires_grad = True
                param_list.append(param)
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        if isinstance(m, nn.BatchNorm2d):
            m_num += 1
            m.train()
            for param in m.parameters():
                param.requires_grad = True
                param_list.append(param)

            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    for i in range(len(network.module.final_layer)):
        m_num = 0
        for m in network.module.final_layer[i].modules():
            if isinstance(m, nn.Conv2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            if isinstance(m, nn.BatchNorm2d):
                m_num += 1
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
                    param_list.append(param)

                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)

    param_list.extend(network.module.classifier.parameters())

    # optim = SGD(network.module.classifier[-1].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optim = SGD(param_list, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for e in range(nepoch):
        total_loss = 0.0
        total_aux_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for i, (inputs, targets) in enumerate(loader, start=1):
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss = criterion(outputs / temperature, targets)
            aux_loss = get_aux_loss(outputs / temperature, targets, _increments)

            total_loss += loss
            total_aux_loss += aux_loss

            if use_aux:
                loss+=aux_loss*aux_loss_weight
            loss.backward()
            optim.step()

            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (e, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning ce loss %.3f aux loss %.3f acc %.3f" %
                        (e, total_loss.item()/i, total_aux_loss.item()/i, total_correct.item() / total_count))
    return network






def finetune_last_layer_ens2(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()
    optim = SGD(network.module.classifier[-1].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['raw_logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss = criterion(outputs / temperature, targets)
            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
    return network






def extract_features(model, loader):
    targets, features = [], []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in loader:
            _inputs = _inputs.cuda()
            _targets = _targets.numpy()
            _features = model(_inputs)['feature'].detach().cpu().numpy()
            features.append(_features)
            targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def calc_class_mean(network, loader, class_idx, metric):
    EPSILON = 1e-8
    features, targets = extract_features(network, loader)
    # norm_feats = features/(np.linalg.norm(features, axis=1)[:,np.newaxis]+EPSILON)
    # examplar_mean = norm_feats.mean(axis=0)
    examplar_mean = features.mean(axis=0)
    if metric == "cosine" or metric == "weight":
        examplar_mean /= (np.linalg.norm(examplar_mean) + EPSILON)
    return examplar_mean


def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means
