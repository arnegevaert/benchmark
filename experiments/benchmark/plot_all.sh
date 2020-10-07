source ../../venv/bin/activate

python plot_benchmark.py -d out/bm/MNIST
python plot_benchmark.py -d out/bm/CIFAR10
python plot_benchmark.py -d out/bm/ImageNette

python inter_method_reliability.py -d out/bm/MNIST -o out/imr/mnist.png
python inter_method_reliability.py -d out/bm/CIFAR10 -o out/imr/cifar.png
python inter_method_reliability.py -d out/bm/ImageNette -o out/imr/imagenette.png

python internal_consistency_reliability.py -d out/bm/MNIST -o out/icr/mnist.png
python internal_consistency_reliability.py -d out/bm/CIFAR10 -o out/icr/cifar.png
python internal_consistency_reliability.py -d out/bm/ImageNette -o out/icr/imagenette.png
