#!/bin/bash

batchsize="16"
split="train"
model="ViT-bigG-14"

if [ $model == 'ViT-B-32' ]; then
    pretrained="laion2b_s34b_b79k"
elif [ $model == 'ViT-g-14' ]; then
    pretrained="laion2b_s12b_b42k"
elif [ $model == 'ViT-bigG-14' ]; then
    pretrained="laion2b_s39b_b160k"
fi

python src/preprocess/augment.py --batchsize $batchsize --split $split --model $model --pretrained $pretrained