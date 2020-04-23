from util.models import MNISTCNN, CifarResNet
from util.datasets import MNIST, Cifar


DATASET_MODELS = {
    "MNIST": {"constructor": MNIST, "models": {"CNN": MNISTCNN}},
    "CIFAR10": {"constructor": lambda **kwargs: Cifar(version="cifar10", **kwargs),
                "models": {"resnet20": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet20", **kwargs),
                           "resnet32": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet32", **kwargs),
                           "resnet44": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet44", **kwargs),
                           "resnet56": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet56", **kwargs)}},
    "CIFAR100": {"constructor": lambda **kwargs: Cifar(version="cifar100", **kwargs, **kwargs),
                 "models": {"resnet20": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet20", **kwargs),
                            "resnet32": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet32", **kwargs),
                            "resnet44": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet44", **kwargs),
                            "resnet56": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet56", **kwargs)}}
}
