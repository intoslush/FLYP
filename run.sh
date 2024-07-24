#!/bin/bash

# 创建软链接
# ln -s ./data/caltech-101 ./datasets/data/caltech-101

# 运行数据创建脚本
# python datacreation_scripts/caltech101.py
conda activate flyp
export PYTHONPATH="$PYTHONPATH:$PWD"
# 运行主程序
# python src/main.py \
#   --train-dataset=Caltech101Val \
#   --epochs=2 \
#   --lr=1e-5 \
#   --wd=0.0 \
#   --batch-size=256 \
#   --model=ViT-B/16 \
#   --warmup_length=500 \
#   --eval-datasets=Caltech101Val,Caltech101Test \
#   --template=caltech101_template \
#   --save=./checkpoints/ \
#   --data-location=./datasets/data/ \
#   --ft_data="./datasets/csv/caltech101/train.csv" \
#   --csv-img-key=filepath \
#   --csv-caption-key=title \
#   --exp_name=caltech101/flyp_loss

python src/main.py \
  --train-dataset=Caltech101Val \
  --epochs=2 \
  --lr=1e-5 \
  --wd=0.2 \
  --batch-size=256 \
  --model=ViT-B/32 \
  --warmup_length 0 \
  --eval-datasets=Caltech101Val,Caltech101Test \
  --template=caltech101_template \
  --save=./checkpoints/ \
  --data-location=./datasets/data/ \
  --ft_data="./datasets/csv/caltech101/train.csv" \
  --csv-img-key=filepath \
  --csv-caption-key=title \
  --exp_name=caltech101/"flyp_loss_4shot" \
  --k=4