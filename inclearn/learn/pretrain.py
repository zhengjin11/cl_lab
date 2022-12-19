import os.path as osp

import torch
import torch.nn.functional as F
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


def _compute_loss(cfg, logits, targets, device):

    if cfg["train_head"] == "sigmoid":
        n_classes = cfg["start_class"]
        onehot_targets = utils.to_onehot(targets, n_classes).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
    elif cfg["train_head"] == "softmax":
        loss = F.cross_entropy(logits, targets)
    else:
        raise ValueError()

    return loss


def train(cfg, model, optimizer, device, train_loader):
    _loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.train()

    # # find lr ================================================================================
    # import math
    # beta = 0.98
    # num = len(train_loader) - 1
    # init_value = 1e-4
    # final_value = 100
    # mult = (final_value / init_value) ** (1 / num)
    # lr = init_value
    # optimizer.param_groups[0]['lr'] = lr
    # avg_loss = 0.
    # best_loss = 0.
    # batch_num = 0
    # losses = []
    # log_lrs = []
    # # find lr ================================================================================

    for i, (inputs, targets) in enumerate(train_loader, start=1):
        # assert torch.isnan(inputs).sum().item() == 0
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model._parallel_network(inputs)['logit']
        if accu is not None:
            accu.add(logits.detach(), targets)

        loss = _compute_loss(cfg, logits, targets, device)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

        loss.backward()
        optimizer.step()
        _loss += loss

        # # find lr ================================================================================
        # # Compute the smoothed loss
        # batch_num += 1
        # print(f"batch {batch_num}")
        # avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        # smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # # Stop if the loss is exploding
        # if batch_num > 1 and smoothed_loss > 1000 * best_loss:
        #     break
        # # Record the best loss
        # if smoothed_loss < best_loss or batch_num == 1:
        #     best_loss = smoothed_loss
        # losses.append(smoothed_loss)
        # log_lrs.append(math.log10(lr))
        # # Update the lr for the next step
        # lr *= mult
        # optimizer.param_groups[0]['lr'] = lr
        # # find lr ================================================================================

    # # find lr ================================================================================
    # # finish
    # print("Y : ", losses)
    # print("X : ", log_lrs)
    # import pdb
    # pdb.set_trace()
    # # find lr ================================================================================

    return (
        round(_loss.item() / i, 3),
        round(accu.value()[0], 3),
    )


def test(cfg, model, device, test_loader):
    _loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader, start=1):
            # assert torch.isnan(inputs).sum().item() == 0
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model._parallel_network(inputs)['logit']
            if accu is not None:
                accu.add(logits.detach(), targets)
            loss = _compute_loss(cfg, logits, targets, device)
            if torch.isnan(loss):
                import pdb
                pdb.set_trace()

            _loss = _loss + loss
    return round(_loss.item() / i, 3), round(accu.value()[0], 3)


def pretrain(cfg, ex, model, device, train_loader, test_loader, model_path):
    ex.logger.info(f"nb Train {len(train_loader.dataset)} Eval {len(test_loader.dataset)}")
    optimizer = torch.optim.SGD(model._network.parameters(),
                                lr=cfg["pretrain"]["lr"],
                                momentum=0.9,
                                weight_decay=cfg["pretrain"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     cfg["pretrain"]["scheduling"],
                                                     gamma=cfg["pretrain"]["lr_decay"])
    test_loss, test_acc = float("nan"), float("nan")
    for e in range(cfg["pretrain"]["epochs"]):
        train_loss, train_acc = train(cfg, model, optimizer, device, train_loader)
        if e % 5 == 0:
            test_loss, test_acc = test(cfg, model, device, test_loader)
            ex.logger.info(
                "Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} | Eval loss: {}, Accu {}".format(
                    cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc, test_loss, test_acc))
        else:
            ex.logger.info("Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} ".format(
                cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc))
        scheduler.step()
    if hasattr(model._network, "module"):
        torch.save(model._network.module.state_dict(), model_path)
    else:
        torch.save(model._network.state_dict(), model_path)
