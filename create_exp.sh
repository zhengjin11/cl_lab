#!/usr/bin/env bash

mkdir -p ./exps/ensmodel1/cifar100/base50_inc10/

rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox \
      ./codes/base/ \
      ./exps/ensmodel1/cifar100/base50_inc10/