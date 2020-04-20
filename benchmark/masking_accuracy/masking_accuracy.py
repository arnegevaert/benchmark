import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def get_logits(self, x):
        relu = nn.ReLU()
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.dropout3(x)
        return self.fc3(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        return F.softmax(logits, dim=1)


if __name__ == '__main__':
    from benchmark.masking_accuracy import MaskedNeuralNetwork, train_masked_network, MaskedDataset
    from util.datasets import Cifar
    import pickle as pkl

    # Initialization
    DATASET = "CIFAR10"
    BATCH_SIZE = 64
    N_BATCHES = 16
    MEDIAN_VALUE = -.788235
    METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
               "GuidedBackprop", "Deconvolution", "Random"]  # , "Occlusion"]
    MODEL_LOC = "../../data/models/cifar10_masked_cnn.pkl"

    # Create masked model and masked dataset
    model = MaskedNeuralNetwork(sample_shape=(3, 32, 32), mask_radius=5, mask_value=0., net=Net())
    orig_ds = Cifar(version="cifar10", batch_size=BATCH_SIZE, shuffle=False,
                    download=False, data_location="../../data/CIFAR10")
    masked_ds = MaskedDataset(orig_ds.get_train_loader(), orig_ds.get_test_loader(), radius=5,
                              mask_value=0., med_of_min=MEDIAN_VALUE)

    # Train the model on synthetic labels
    train_masked_network(model, masked_ds, lr=1.0, gamma=0.7, epochs=10)
    pkl.dump(model, open(MODEL_LOC, "wb"))
