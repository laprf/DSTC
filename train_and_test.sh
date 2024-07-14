#!/usr/bin/bash

BACKBONE='resnet' # resnet, PVTV2, swin
DATA_ROOT='data/'

if [ $BACKBONE != 'resnet' ] && [ $BACKBONE != 'PVTV2' ] && [ $BACKBONE != 'swin' ]; then
    echo "backbone only supports ResNet, PVTV2, and Swin!"
    exit 1
fi

echo "Using $BACKBONE as backbone, now training..."
CUDA_VISIBLE_DEVICES=1 python train.py --config=models/yamls/$BACKBONE.yaml \
                                       --exp_name=$BACKBONE backbone \
                                       --log_path=logs/$BACKBONE.log \
                                       --data_root=$DATA_ROOT

echo "Done training on $BACKBONE, now testing..."
CUDA_VISIBLE_DEVICES=1 python test.py --config=models/yamls/$BACKBONE.yaml \
                                      --log_path=logs/$BACKBONE.log \
                                      --data_root=$DATA_ROOT \
                                      --pretrained_model=DataStorage/$BACKBONE/best_model.pth