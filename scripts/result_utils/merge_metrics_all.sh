source ../../venv/bin/activate

for ds in mnist cifar10 imagenet; do
  python merge_metrics.py ../../out/maxsens/$ds.h5 ../../out/merged/$ds.h5 ../../out/merged_maxsens/$ds.h5
done
