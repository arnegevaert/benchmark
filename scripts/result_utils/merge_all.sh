source ../../venv/bin/activate

for ds in mnist fashionmnist cifar10 cifar100 svhn imagenet caltech places; do
  python merge_methods.py ../../out/lime/$ds.h5 ../../out/no_lime/$ds.h5 ../../out/merged/$ds.h5
done