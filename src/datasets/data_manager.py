# Manages datasets, their derived variants and trained models
class DataManager:
    def __init__(self, data_root):
        self.data_root = data_root
        self.datasets = {}
        # TODO traverse root directory and derive which datasets are available
