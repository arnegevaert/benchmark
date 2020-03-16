from models import MNISTCNN, CifarResNet
from datasets import MNIST, Cifar


DATASET_MODELS = {
    "MNIST": {"constructor": MNIST, "models": {"CNN": MNISTCNN}},
    "CIFAR10": {"constructor": lambda **kwargs: Cifar(version="cifar10", **kwargs),
                "models": {"resnet20": lambda: CifarResNet(dataset="cifar10", resnet="resnet20"),
                           "resnet32": lambda: CifarResNet(dataset="cifar10", resnet="resnet32"),
                           "resnet44": lambda: CifarResNet(dataset="cifar10", resnet="resnet44"),
                           "resnet56": lambda: CifarResNet(dataset="cifar10", resnet="resnet56")}},
    "CIFAR100": {"constructor": lambda **kwargs: Cifar(version="cifar100", **kwargs),
                 "models": {"resnet20": lambda: CifarResNet(dataset="cifar100", resnet="resnet20"),
                            "resnet32": lambda: CifarResNet(dataset="cifar100", resnet="resnet32"),
                            "resnet44": lambda: CifarResNet(dataset="cifar100", resnet="resnet44"),
                            "resnet56": lambda: CifarResNet(dataset="cifar100", resnet="resnet56")}}
}
