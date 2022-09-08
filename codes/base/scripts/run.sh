#!/usr/bin/env bash
# methd_dataset_setting_seed
name='sepmodel0_cifar100_base50_inc10_intra0.6_inter1.4_seed3'
class_order=3
comments='None'
expid='1'
export CUDA_VISIBLE_DEVICES=2

python -m main train with "./configs/${expid}.yaml" \
    exp.name="${name}" \
    exp.savedir="./logs/" \
    exp.ckptdir="./logs/" \
    exp.tensorboard_dir="./tensorboard/" \
    trial=${class_order} \
    weight_normalization=True \
    --name="${name}" \
    -D \
    -p \
    --force \
    #--mongo_db=10.10.10.100:30620:debug

