#!/usr/bin/bash

BACKBONE='resnet' # resnet, PVTV2, swin
DATA_ROOT='data/'

if [ $BACKBONE != 'resnet' ] && [ $BACKBONE != 'PVTV2' ] && [ $BACKBONE != 'swin' ]; then
    echo "backbone only supports ResNet, PVTV2, and Swin!"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python test.py --config=models/yamls/$BACKBONE.yaml \
                                      --log_path=logs/$BACKBONE.log \
                                      --data_root=$DATA_ROOT \
                                      --pretrained_model=DataStorage/$BACKBONE/best_model.pth
