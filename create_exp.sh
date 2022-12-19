#!/usr/bin/env bash

mkdir -p ./exps/expnet4_mulcls/cifar100/10steps_20_per_class/

rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox \
      ./codes/base/ \
      ./exps/expnet4_mulcls/cifar100/10steps_20_per_class/


