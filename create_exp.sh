#!/usr/bin/env bash

mkdir -p ./exps/expnet4_mulcls/cifar100/base50_inc5/

rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox \
      ./codes/base/ \
      -p ./exps/expnet4_mulcls/cifar100/base50_inc5/


