#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate benchmark


for ds in Caltech256 Places365; do
    for model in resnet18 resnet50; do
        echo $ds $model
        python -W ignore precomp_attrs.py config/suite.yaml ../../out/attrs/$ds\_$model -d $ds -m $model -b 32 -c -o ../../out/$ds\_$model.h5 --seed 42
    done
done

