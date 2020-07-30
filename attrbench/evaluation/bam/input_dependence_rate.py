from attrbench.datasets import BAMDataset
from typing import Iterable, Callable, Dict


def input_dependence_rate(dataset: BAMDataset, model: Callable, methods: Dict[str, Callable],
                          device: str):
    """
    Input dependence rate:
    Percentage of correctly classified inputs with object overlay by model trained on scene labels,
    where region of object overlay is attributed less than that same region when the overlay
    is removed. 1 - IDR can be interpreted as "false positive rate": rate of explanations
    that assign higher importance to less important features.
    """
    pass


if __name__ == "__main__":
    ds = BAMDataset(4, "../../data/bam")