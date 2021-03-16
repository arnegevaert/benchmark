#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate benchmark


echo "Places365 resnet50"
python -W ignore compute_attrs.py config/methods.yaml -d Places365 -m resnet50 -b 32 -n 256 -c -o ../../out/attrs/places365_resnet50 --seed 42
