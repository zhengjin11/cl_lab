import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier



class BasicNet_ens0(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens0, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier



class BasicNet_ens1(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens1, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        # self.classifier = None
        self.classifier = nn.ModuleList()
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        # if self.der:
        #     features = [convnet(x) for convnet in self.convnets]
        #     features = torch.cat(features, 1)
        # else:
        #     features = self.convnet(x)
        #
        # logits = self.classifier(features)

        # aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None

        inner_softmax_vector = []
        raw_logits = []
        features = []

        for i in range(len(self.convnets)):
            temp_features = self.convnets[i](x)
            temp_logits = self.classifier[i](temp_features)

            features.append(temp_features)
            raw_logits.append(temp_logits)

            # inner task softmax
            temp_logits = F.softmax(temp_logits, dim=1)
            inner_softmax_vector.append(temp_logits)

        features = torch.cat(features, 1)
        inner_softmax_vector = torch.cat(inner_softmax_vector, 1)
        raw_logits = torch.cat(raw_logits, 1)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': raw_logits, 'aux_logit': aux_logits, 'inner_softmax_vector': inner_softmax_vector}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        # if self.classifier is not None:
        #     weight = copy.deepcopy(self.classifier.weight.data)

        # fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)


        # if self.classifier is not None and self.reuse_oldfc:
        #     fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        # del self.classifier
        # self.classifier = fc

        fc = self._gen_classifier(self.out_dim, n_classes)
        self.classifier.append(fc)

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        # else:
        #     aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier





class BasicNet_ens2(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens2, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        # self.classifier = None
        self.classifier = nn.ModuleList()
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        # if self.der:
        #     features = [convnet(x) for convnet in self.convnets]
        #     features = torch.cat(features, 1)
        # else:
        #     features = self.convnet(x)
        #
        # logits = self.classifier(features)

        # aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None

        logits = []
        raw_logits = []
        features = []
        cur_logits = None


        for i in range(len(self.convnets)):
            temp_features = self.convnets[i](x)
            temp_logits = self.classifier[i](temp_features)

            features.append(temp_features)

            if i==len(self.convnets)-1:
                cur_logits = temp_logits

            raw_logits.append(temp_logits)
            # inner task softmax
            temp_logits = F.softmax(temp_logits, dim=1)
            logits.append(temp_logits)

        features = torch.cat(features, 1)
        logits = torch.cat(logits, 1)
        raw_logits = torch.cat(raw_logits, 1)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if self.aux_classifier is not None else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits, 'cur_logit':cur_logits, 'raw_logit':raw_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        # if self.classifier is not None:
        #     weight = copy.deepcopy(self.classifier.weight.data)

        # fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)


        # if self.classifier is not None and self.reuse_oldfc:
        #     fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        # del self.classifier
        # self.classifier = fc

        fc = self._gen_classifier(self.out_dim, n_classes)
        self.classifier.append(fc)

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
        else:
            aux_fc = None

            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        # else:
        #     aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier





class BasicNet_sep0(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_sep0, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier



class BasicNet_ens6(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens6, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        # if self.der:
        #     features = [convnet(x) for convnet in self.convnets]
        #     features = torch.cat(features, 1)
        # else:
        #     features = self.convnet(x)

        origin_input = x
        features = []
        temp_features = []

        for i in range(len(self.convnets)):
            if i==0:
                x = self.convnets[i].conv1(x)
                temp_features.append(x)
                x = self.convnets[i].layer1(x)
                temp_features.append(x)
                x = self.convnets[i].layer2(x)
                # temp_features.append(x)
                x = self.convnets[i].layer3(x)
                # temp_features.append(x)
                x = self.convnets[i].layer4(x)
                x = self.convnets[i].avgpool(x)
                x = x.view(x.size(0), -1)
                features.append(x)
            else:
                # import pdb
                # pdb.set_trace()
                idx = 0
                x = self.convnets[i].conv1(origin_input)
                x = x + temp_features[idx]
                temp_features[idx] = x.clone()
                x= x/(i+1)
                idx += 1

                x = self.convnets[i].layer1(x)
                x = x + temp_features[idx]
                temp_features[idx] = x.clone()
                x= x/(i+1)
                idx += 1

                x = self.convnets[i].layer2(x)
                # x = x + temp_features[idx]
                # temp_features[idx] = x.clone()
                # x= x/(i+1)
                # idx += 1

                x = self.convnets[i].layer3(x)
                # x = x + temp_features[idx]
                # temp_features[idx] = x.clone()
                # x= x/(i+1)
                # idx += 1

                x = self.convnets[i].layer4(x)
                x = self.convnets[i].avgpool(x)
                x = x.view(x.size(0), -1)
                features.append(x)

        features = torch.cat(features, 1)
        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier



class BasicNet_ens7(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens7, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.final_layer = nn.ModuleList()
            self.out_dim = self.convnet.out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            if len(self.final_layer)==0:
                features = self.convnet(x)
            else:
                features = []

                x = self.convnet.conv1(x)
                x = self.convnet.layer1(x)
                x = self.convnet.layer2(x)
                temp_feature = x

                x = self.convnet.layer3(x)
                x = self.convnet.layer4(x)
                x = self.convnet.avgpool(x)
                x = x.view(x.size(0), -1)
                features.append(x)

                for i in range(len(self.final_layer)):
                    x = self.final_layer[i](temp_feature)
                    x = self.convnet.avgpool(x)
                    x = x.view(x.size(0), -1)
                    features.append(x)

                features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * (len(self.final_layer)+1)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_final_layer_ = []
            new_layer3 = copy.deepcopy(self.convnet.layer3)
            new_final_layer_.append(new_layer3)
            new_layer4 = copy.deepcopy(self.convnet.layer4)
            new_final_layer_.append(new_layer4)
            new_final_layer = nn.Sequential(*new_final_layer_)

            m_num = 0
            for m in new_final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    m_num += 1
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                if isinstance(m, nn.BatchNorm2d):
                    m_num += 1
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)

            self.final_layer.append(new_final_layer)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc


    # def _add_classes_multi_fc(self, n_classes):
    #     if self.ntask > 1:
    #         new_final_layer = copy.deepcopy(self.convnet.layer4)
    #         m_num = 0
    #
    #         for m in new_final_layer.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 m_num += 1
    #                 torch.nn.init.xavier_normal_(m.weight.data)
    #                 if m.bias is not None:
    #                     torch.nn.init.constant_(m.bias.data, 0.0)
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m_num += 1
    #                 m.weight.data.fill_(1.0)
    #                 m.bias.data.fill_(0.0)
    #
    #         self.final_layer.append(new_final_layer)
    #
    #     if self.classifier is not None:
    #         weight = copy.deepcopy(self.classifier.weight.data)
    #
    #     fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)
    #
    #     if self.classifier is not None and self.reuse_oldfc:
    #         fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight
    #     del self.classifier
    #     self.classifier = fc
    #
    #     if self.aux_nplus1:
    #         aux_fc = self._gen_classifier(self.out_dim, n_classes)
    #         # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
    #         # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
    #     else:
    #         aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
    #     del self.aux_classifier
    #     self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier





class BasicNet_ens8(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_ens8, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.final_layer = nn.ModuleList()
            self.out_dim = self.convnet.out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

        # conv1_num = sum(p.numel() for p in self.convnet.conv1.parameters())
        # layer1_num = sum(p.numel() for p in self.convnet.layer1.parameters())
        # layer2_num = sum(p.numel() for p in self.convnet.layer2.parameters())
        # layer3_num = sum(p.numel() for p in self.convnet.layer3.parameters())
        # layer4_num = sum(p.numel() for p in self.convnet.layer4.parameters())
        # 
        # total_sum = conv1_num+layer1_num+layer2_num+layer3_num+layer4_num
        # 
        # print(f"conv1 : {conv1_num} , {(conv1_num/total_sum)*100}")
        # print(f"layer1 : {layer1_num} , {(layer1_num / total_sum) * 100}")
        # print(f"layer2 : {layer2_num} , {(layer2_num / total_sum) * 100}")
        # print(f"layer3 : {layer3_num} , {(layer3_num / total_sum) * 100}")
        # print(f"layer4 : {layer4_num} , {(layer4_num / total_sum) * 100}")
        # 
        # import pdb
        # pdb.set_trace()

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            if len(self.final_layer)==0:
                features = self.convnet(x)
            else:
                features = []

                x = self.convnet.conv1(x)
                x = self.convnet.layer1(x)
                x = self.convnet.layer2(x)
                x = self.convnet.layer3(x)
                temp_feature = x

                x = self.convnet.layer4(x)
                x = self.convnet.avgpool(x)
                x = x.view(x.size(0), -1)
                features.append(x)

                for i in range(len(self.final_layer)):
                    x = self.final_layer[i](temp_feature)
                    x = self.convnet.avgpool(x)
                    x = x.view(x.size(0), -1)
                    features.append(x)

                features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * (len(self.final_layer)+1)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_final_layer_ = []
            # new_layer3 = copy.deepcopy(self.convnet.layer3)
            # new_final_layer_.append(new_layer3)
            new_layer4 = copy.deepcopy(self.convnet.layer4)
            new_final_layer_.append(new_layer4)
            new_final_layer = nn.Sequential(*new_final_layer_)

            m_num = 0
            for m in new_final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    m_num += 1
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                if isinstance(m, nn.BatchNorm2d):
                    m_num += 1
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)

            self.final_layer.append(new_final_layer)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc


    # def _add_classes_multi_fc(self, n_classes):
    #     if self.ntask > 1:
    #         new_final_layer = copy.deepcopy(self.convnet.layer4)
    #         m_num = 0
    #
    #         for m in new_final_layer.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 m_num += 1
    #                 torch.nn.init.xavier_normal_(m.weight.data)
    #                 if m.bias is not None:
    #                     torch.nn.init.constant_(m.bias.data, 0.0)
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m_num += 1
    #                 m.weight.data.fill_(1.0)
    #                 m.bias.data.fill_(0.0)
    #
    #         self.final_layer.append(new_final_layer)
    #
    #     if self.classifier is not None:
    #         weight = copy.deepcopy(self.classifier.weight.data)
    #
    #     fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)
    #
    #     if self.classifier is not None and self.reuse_oldfc:
    #         fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight
    #     del self.classifier
    #     self.classifier = fc
    #
    #     if self.aux_nplus1:
    #         aux_fc = self._gen_classifier(self.out_dim, n_classes)
    #         # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
    #         # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
    #     else:
    #         aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
    #     del self.aux_classifier
    #     self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier




class BasicNet_exp1(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_exp1, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.final_layer = nn.ModuleList()
            self.out_dim = self.convnet.out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None
        self.old_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            if len(self.final_layer)==0:
                task_id = 0
                features = self.convnet(x, task_id)
            else:
                features = []
                input = x

                for task_id in range(self.ntask):
                    x = self.convnet.conv1(input, task_id)
                    x = self.convnet.layer1(x, task_id)
                    x = self.convnet.layer2(x, task_id)

                    if task_id == 0:
                        x = self.convnet.layer3(x)
                        x = self.convnet.layer4(x)
                        x = self.convnet.avgpool(x)
                        x = x.view(x.size(0), -1)
                        features.append(x)
                    else:
                        x = self.final_layer[task_id-1](x)
                        x = self.convnet.avgpool(x)
                        x = x.view(x.size(0), -1)
                        features.append(x)

                features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        old_cls_logits = self.old_classifier(features[:, :features.shape[1]-self.out_dim]) if features.shape[1] > self.out_dim else None

        if aux_logits is not None and old_cls_logits is not None:
            aux_p_vec = F.softmax(aux_logits, dim=1)
            old_cls_p_vec = F.softmax(old_cls_logits, dim=1)
            mix_p_vec = torch.cat([old_cls_p_vec, aux_p_vec], 1)
        else:
            mix_p_vec = None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits, 'mix_p_vec': mix_p_vec}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * (len(self.final_layer)+1)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1
        print(self.convnet.conv1.module_list)

        if self.ntask>1:
            self.convnet.add_new_task_bn()
            self.to(self.device)

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_final_layer_ = []
            new_layer3 = copy.deepcopy(self.convnet.layer3)
            new_final_layer_.append(new_layer3)
            new_layer4 = copy.deepcopy(self.convnet.layer4)
            new_final_layer_.append(new_layer4)
            new_final_layer = nn.Sequential(*new_final_layer_)

            m_num = 0
            for m in new_final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    m_num += 1
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                if isinstance(m, nn.BatchNorm2d):
                    m_num += 1
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)

            self.final_layer.append(new_final_layer)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight

        self.old_classifier = copy.deepcopy(self.classifier)
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier





class BasicNet_exp2(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_exp2, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.final_layer = nn.ModuleList()
            self.out_dim = self.convnet.out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None
        self.old_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            if len(self.final_layer)==0:
                task_id = 0
                features = self.convnet(x, task_id)
            else:
                features = []
                input = x

                for task_id in range(self.ntask):
                    x = self.convnet.conv1(input, task_id)
                    x = self.convnet.layer1(x, task_id)
                    x = self.convnet.layer2(x, task_id)
                    x = self.convnet.layer3(x, task_id)

                    if task_id == 0:
                        x = self.convnet.layer4(x)
                        x = self.convnet.avgpool(x)
                        x = x.view(x.size(0), -1)
                        features.append(x)
                    else:
                        x = self.final_layer[task_id-1](x)
                        x = self.convnet.avgpool(x)
                        x = x.view(x.size(0), -1)
                        features.append(x)

                features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        old_cls_logits = self.old_classifier(features[:, :features.shape[1]-self.out_dim]) if features.shape[1] > self.out_dim else None

        if aux_logits is not None and old_cls_logits is not None:
            aux_p_vec = F.softmax(aux_logits, dim=1)
            old_cls_p_vec = F.softmax(old_cls_logits, dim=1)
            mix_p_vec = torch.cat([old_cls_p_vec, aux_p_vec], 1)
        else:
            mix_p_vec = None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits, 'mix_p_vec': mix_p_vec}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * (len(self.final_layer)+1)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1
        print(self.convnet.conv1.module_list)

        if self.ntask>1:
            self.convnet.add_new_task_bn()
            self.to(self.device)

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_final_layer_ = []
            # new_layer3 = copy.deepcopy(self.convnet.layer3)
            # new_final_layer_.append(new_layer3)
            new_layer4 = copy.deepcopy(self.convnet.layer4)
            new_final_layer_.append(new_layer4)
            new_final_layer = nn.Sequential(*new_final_layer_)

            m_num = 0
            for m in new_final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    m_num += 1
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                if isinstance(m, nn.BatchNorm2d):
                    m_num += 1
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)

            self.final_layer.append(new_final_layer)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * (len(self.final_layer)+1), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight

        self.old_classifier = copy.deepcopy(self.classifier)
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier





class BasicNet_exp3(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet_exp3, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.final_layer = nn.ModuleList()
            self.out_dim = self.convnet.out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = []
            input = x

            for task_id in range(self.ntask):
                x = self.convnet.conv1(input, task_id)
                x = self.convnet.layer1(x, task_id)
                x = self.convnet.layer2(x, task_id)
                x = self.convnet.layer3(x, task_id)
                x = self.convnet.layer4(x, task_id)
                x = self.convnet.avgpool(x)
                x = x.view(x.size(0), -1)
                features.append(x)

            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * (self.ntask)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1
        print(self.convnet.conv1.module_list)

        if self.ntask>1:
            self.convnet.add_new_task_bn()
            self.to(self.device)

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            pass
            # new_final_layer_ = []
            # # new_layer3 = copy.deepcopy(self.convnet.layer3)
            # # new_final_layer_.append(new_layer3)
            # new_layer4 = copy.deepcopy(self.convnet.layer4)
            # new_final_layer_.append(new_layer4)
            # new_final_layer = nn.Sequential(*new_final_layer_)
            #
            # m_num = 0
            # for m in new_final_layer.modules():
            #     if isinstance(m, nn.Conv2d):
            #         m_num += 1
            #         torch.nn.init.xavier_normal_(m.weight.data)
            #         if m.bias is not None:
            #             torch.nn.init.constant_(m.bias.data, 0.0)
            #     if isinstance(m, nn.BatchNorm2d):
            #         m_num += 1
            #         m.weight.data.fill_(1.0)
            #         m.bias.data.fill_(0.0)
            #
            # self.final_layer.append(new_final_layer)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * (self.ntask), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.final_layer))] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_fc_classifier(self.out_dim, n_classes)
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_fc_classifier(self, in_features, n_classes):

        classifier = nn.Linear(in_features, n_classes, bias=True).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        nn.init.constant_(classifier.bias, 0.0)

        return classifier


