class ImageDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mask_value = 0.

    def get_train_loader(self):
        raise NotImplementedError

    def get_test_loader(self):
        raise NotImplementedError

    def get_sample_shape(self):
        raise NotImplementedError

    def get_batch_size(self):
        return self.batch_size
