import numpy as np
import random
import time
import math
import os
from copy import deepcopy
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from scipy.special import softmax
import sklearn.metrics as sk

from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer,deep_finetune_last_layer_ens7


# description


# Constants
EPSILON = 1e-8

aux_loss_part2_weight = 0.5
old_cls_loss_part2_weight = 0.5

use_oe_finetune = False
oe_finetune_loss_weight = 0.5

recall_level_default = 0.95

class ExpNet4_mulcls(IncrementalLearner):
    def __init__(self, cfg, trial_i, _run, ex, tensorboard, inc_dataset):

        super().__init__()
        print("create expnet4_mulcls !!!")

        self._cfg = cfg
        self._device = cfg['device']
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self._trial_i = trial_i  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]

        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.BasicNet_exp4(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )
        self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        if self._der:
            if self._task == 0:
                self._parallel_network.train()
            elif self._task >= 1:
                self._parallel_network.train()
                self._parallel_network.module.exp_layer[-1].train()

                self._parallel_network.module.convnet.eval()
                self._parallel_network.module.convnet.freeze_old_task_bn()
                self._parallel_network.module.convnet.enable_new_task_bn()
                for i in range(self._task-1):
                    self._parallel_network.module.exp_layer[i].eval()
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        self._ex.logger.info(f"Begin step {taski}")

        # Update Task info
        self._task = taski
        self._n_classes += self._task_size

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        if self._der and self._task > 0:
            for p in self._parallel_network.module.convnet.parameters():
                p.requires_grad = False
            self._parallel_network.module.convnet.freeze_old_task_bn()
            self._parallel_network.module.convnet.enable_new_task_bn()

            for i in range(self._task-1):
                for p in self._parallel_network.module.exp_layer[i].parameters():
                    p.requires_grad = False

            for p in self._parallel_network.module.exp_layer[-1].parameters():
                p.requires_grad = True

        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr, weight_decay)

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)

    def _train_task(self, train_loader, val_loader):
        self._ex.logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "Initial trainset")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Initial trainset")

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            _loss, _loss_aux = 0.0, 0.0
            _loss_old_cls = 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()

            for i, (inputs, targets) in enumerate(train_loader, start=1):
                self.train()
                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_aux, loss_old_cls = self._forward_loss(
                    inputs,
                    targets,
                    old_classes,
                    new_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )

                if self._cfg["use_aux_cls"] and self._task > 0:
                    loss = loss_ce + loss_aux + loss_old_cls
                    # loss = loss_ce + loss_aux
                else:
                    loss = loss_ce

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)

                _loss += loss_ce
                _loss_aux += loss_aux
                _loss_old_cls += loss_old_cls
            _loss = _loss.item()
            _loss_aux = _loss_aux.item()
            _loss_old_cls = _loss_old_cls.item()
            if not self._warmup:
                self._scheduler.step()
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {}, Oldcls loss: {}, Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss / i, 3),
                    round(_loss_aux / i, 3),
                    round(_loss_old_cls / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                    round(train_old_accu.value()[0], 3),
                ))
            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, inputs, targets, old_classes, new_classes, accu=None, new_accu=None, old_accu=None):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
            # accu.add(logits.detach(), targets.cpu().numpy())
        # if new_accu is not None:
        #     new_accu.add(logits[new_classes].detach(), targets[new_classes].cpu().numpy())
        # if old_accu is not None:
        #     old_accu.add(logits[old_classes].detach(), targets[old_classes].cpu().numpy())
        return self._compute_loss(inputs, targets, outputs, old_classes, new_classes)

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes):
        loss = F.cross_entropy(outputs['logit'], targets)

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            # if self._cfg["aux_n+1"]:
            #     aux_targets[old_classes] = 0
            #     aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            #     import pdb
            #     pdb.set_trace()
            # aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)

            if self._cfg["aux_n+1"]:
                # new class
                if outputs['aux_logit'][new_classes].shape[0] != 0:
                    aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task])
                    aux_loss_part1 = F.cross_entropy(outputs['aux_logit'][new_classes], aux_targets[new_classes])
                else:
                    aux_loss_part1 = 0.0

                # old class
                if outputs['aux_logit'][old_classes].shape[0] != 0:
                    aux_loss_part2 = aux_loss_part2_weight * -(
                            outputs['aux_logit'][old_classes].mean(1) - torch.logsumexp(outputs['aux_logit'][old_classes], dim=1)).mean()
                else:
                    aux_loss_part2 = 0.0
                aux_loss = aux_loss_part1 + aux_loss_part2
        else:
            aux_loss = torch.zeros([1]).cuda()

        if outputs['old_cls_logit'] is not None:
            old_cls_targets = targets.clone()
            if self._cfg["aux_n+1"]:
                # new class
                if outputs['old_cls_logit'][new_classes].shape[0] != 0:
                    old_cls_loss_part2 = old_cls_loss_part2_weight * -(
                            outputs['old_cls_logit'][new_classes].mean(1) - torch.logsumexp(
                        outputs['old_cls_logit'][new_classes], dim=1)).mean()
                else:
                    old_cls_loss_part2 = 0.0

                # old class
                if outputs['old_cls_logit'][old_classes].shape[0] != 0:
                    old_cls_loss_part1 = F.cross_entropy(outputs['old_cls_logit'][old_classes], old_cls_targets[old_classes])
                else:
                    old_cls_loss_part1 = 0.0
                old_cls_loss = old_cls_loss_part1 + old_cls_loss_part2
        else:
            old_cls_loss = torch.zeros([1]).cuda()

        return loss, aux_loss, old_cls_loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()
        # self._ex.logger.info("save model")
        # if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
        #     save_path = os.path.join(os.getcwd(), "ckpts")
        #     torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if (self._cfg["decouple"]['enable'] and taski > 0):
            if self._cfg["decouple"]["fullset"]:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                       inc_dataset.targets_inc,
                                                       mode="balanced_train")

            # finetuning
            self._parallel_network.module.classifier.reset_parameters()
            finetune_last_layer(self._ex.logger,
                                self._parallel_network,
                                train_loader,
                                self._n_classes,
                                nepoch=self._decouple["epochs"],
                                lr=self._decouple["lr"],
                                scheduling=self._decouple["scheduling"],
                                lr_decay=self._decouple["lr_decay"],
                                weight_decay=self._decouple["weight_decay"],
                                loss_type="ce",
                                temperature=self._decouple["temperature"])

            # deep_finetune_last_layer_ens7(self._ex.logger,
            #                     self._parallel_network,
            #                     train_loader,
            #                     self._n_classes,
            #                     nepoch=self._decouple["epochs"],
            #                     lr=self._decouple["lr"],
            #                     scheduling=self._decouple["scheduling"],
            #                     lr_decay=self._decouple["lr_decay"],
            #                     weight_decay=self._decouple["weight_decay"],
            #                     loss_type="ce",
            #                     temperature=self._decouple["temperature"],
            #                     _increments=self._increments,
            #                     use_aux=use_oe_finetune,
            #                     aux_loss_weight=oe_finetune_loss_weight
            #                     )

            network = deepcopy(self._parallel_network)
            if self._cfg["save_ckpt"]:
                save_path = os.path.join(os.getcwd(), "ckpts")
                torch.save(network.cpu().state_dict(), "{}/decouple_step{}.ckpt".format(save_path, self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        network.eval()
        self._ex.logger.info("save model")
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        old_cls_logits, aux_logits = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                ret_dict = self._parallel_network(inputs)
                _old_cls_logits = ret_dict['old_cls_logit']
                _aux_logits = ret_dict['aux_logit']
                old_class_num = sum(self._increments) - self._increments[-1]
                new_class_num = self._increments[-1]

                # case1 1 big cls
                # _preds = ret_dict['logit']

                # case2 2 cls
                # _preds = ret_dict['mix_p_vec']
                # if _preds is None:
                #     _preds = self._parallel_network(inputs)['logit']
                # else:
                #     _preds[:, -self._increments[-1]:] = _preds[:, -self._increments[-1]:] * self._increments[-1]
                #     old_class_num = sum(self._increments) - self._increments[-1]
                #     _preds[:, :old_class_num] = _preds[:, :old_class_num] * old_class_num
                #     pass

                # case3 3 cls
                _preds = ret_dict['logit']
                if self._task > 0:
                    weight_of_new_preds = 1.0
                    _preds = F.softmax(_preds, dim=1)
                    old_expert_preds = F.softmax(_old_cls_logits, dim=1)
                    new_expert_preds = F.softmax(_aux_logits, dim=1)

                    _preds = _preds*(old_class_num+new_class_num)
                    _preds[:, :old_class_num] += old_expert_preds*old_class_num
                    _preds[:, -new_class_num:] += (weight_of_new_preds)*new_expert_preds*new_class_num
                    # _preds = _preds
                    # _preds[:, :old_class_num] += old_expert_preds
                    # _preds[:, -new_class_num:] += (weight_of_new_preds) * new_expert_preds

                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())

                if self._task > 0:
                    old_cls_logits.append(_old_cls_logits.detach().cpu().numpy())
                    aux_logits.append(_aux_logits.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        if self._task > 0:

            self._ex.logger.info(f" {self._increments}, {old_class_num}")

            old_cls_logits = np.concatenate(old_cls_logits, axis=0)
            aux_logits = np.concatenate(aux_logits, axis=0)
            old_classes = targets < (self._n_classes - self._task_size)
            new_classes = targets >= (self._n_classes - self._task_size)

            # 1. eval new expert cls
            self._ex.logger.info("Start eval new expert cls!")
            yraw, ytrue, new_class_logits, old_class_logits = preds[new_classes, -self._task_size:], targets[new_classes], aux_logits[new_classes], aux_logits[old_classes]
            ytrue = ytrue - (self._n_classes - self._task_size)
            ypred = np.argmax(yraw, axis=1)
            acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
            self._ex.logger.info("new expert Val accuracy: {}".format(acc))

                ## get auroc
            auroc_val = get_auroc(new_class_logits, old_class_logits)
            self._ex.logger.info(f"AUROC : {auroc_val}")

                # watch class average logits
            avg_new_logits = np.average(new_class_logits, axis=0)
            avg_new_logits = (avg_new_logits + 1) / 2

            avg_old_logits = np.average(old_class_logits, axis=0)
            avg_old_logits = (avg_old_logits + 1) / 2

            self._ex.logger.info("old class avg logits: \n\n {} \n\n".format(avg_old_logits))
            self._ex.logger.info("new class avg logits: \n\n {} \n\n".format(avg_new_logits))
            self._ex.logger.info("Finish eval new expert cls!")

            # 2. eval old expert cls
            self._ex.logger.info("Start eval old expert cls!")
            yraw, ytrue, new_class_logits, old_class_logits = preds[old_classes, :-self._task_size], targets[old_classes], old_cls_logits[new_classes], old_cls_logits[old_classes]
            ypred = np.argmax(yraw, axis=1)
            acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
            self._ex.logger.info("old expert Val accuracy: {}".format(acc))

            ## get auroc
            auroc_val = get_auroc(old_class_logits, new_class_logits)
            self._ex.logger.info(f"AUROC : {auroc_val}")

            # watch class average logits
            avg_new_logits = np.average(new_class_logits, axis=0)
            avg_new_logits = (avg_new_logits + 1) / 2

            avg_old_logits = np.average(old_class_logits, axis=0)
            avg_old_logits = (avg_old_logits + 1) / 2

            self._ex.logger.info("old class avg logits: \n\n {} \n\n".format(avg_old_logits))
            self._ex.logger.info("new class avg logits: \n\n {} \n\n".format(avg_new_logits))
            self._ex.logger.info("Finish eval new expert cls!")

        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._ex.logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(os.getcwd(), f"ckpts/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._herding_matrix,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
            )
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc


def get_auroc(pos_class_logits, neg_class_logits):

    # get auroc
    smax = softmax(pos_class_logits, axis=1)
    print(smax[0])
    pos_class_score = np.max(smax, axis=1)


    smax = softmax(neg_class_logits, axis=1)
    print(smax[0])
    neg_class_score = np.max(smax, axis=1)

    # import pdb
    # pdb.set_trace()
    auroc_val = get_measures(pos_class_score, neg_class_score)
    return auroc_val
