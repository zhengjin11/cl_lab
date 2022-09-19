#!/usr/bin/env bash

mkdir -p ./exps/der_model/cifar100/10steps_full/

rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox \
      ./codes/base/ \
      ./exps/der_model/cifar100/10steps_full/