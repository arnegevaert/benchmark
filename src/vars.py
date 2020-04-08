from models import MNISTCNN, CifarResNet, MNISTFCNN, AptosDensenet
from datasets import MNIST, Cifar, Aptos


DATASET_MODELS = {
    "MNIST": {"constructor": MNIST,
              "models": {"CNN": MNISTCNN,
                         "FCNN": MNISTFCNN}},
    "CIFAR10": {"constructor": lambda **kwargs: Cifar(version="cifar10", **kwargs),
                "models": {"resnet20": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet20", **kwargs),
                           "resnet32": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet32", **kwargs),
                           "resnet44": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet44", **kwargs),
                           "resnet56": lambda **kwargs: CifarResNet(dataset="cifar10", resnet="resnet56", **kwargs)}},
    "CIFAR100": {"constructor": lambda **kwargs: Cifar(version="cifar100", **kwargs, **kwargs),
                 "models": {"resnet20": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet20", **kwargs),
                            "resnet32": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet32", **kwargs),
                            "resnet44": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet44", **kwargs),
                            "resnet56": lambda **kwargs: CifarResNet(dataset="cifar100", resnet="resnet56", **kwargs)}},
    "APTOS": {"constructor": lambda **kwargs: Aptos(**kwargs),
              "models": {"densenet121": lambda **kwargs: AptosDensenet(densenet="densenet121", **kwargs)}}
}
