from datasets import PerturbedImageDataset, MNIST
from models import MNISTCNN
from itertools import islice
dataset = MNIST(batch_size=4, download=False)
model = MNISTCNN(dataset=dataset)
iterator = islice(iter(dataset.get_test_loader()), 5)
perturbed_dataset = PerturbedImageDataset.generate("../../data", "MNIST_noise", iterator, model)
iterators = perturbed_dataset.get_iterators()

