#!/usr/bin/env bash

mkdir -p ./exps/ensmodel0/cifar100/50steps_20_per_class/

rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox \
      ./codes/base/ \
      ./exps/ensmodel0/cifar100/50steps_20_per_class/