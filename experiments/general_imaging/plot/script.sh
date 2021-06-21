source ../../../venv/bin/activate

for NAME in caltech cifar100 cifar10 fashionmnist imagenet mnist places svhn; do
    python plot.py ../../../out/$NAME.h5 out/$NAME single -p wsp
done
