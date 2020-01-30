import torch
from torchvision import datasets, transforms
from models.scripts.mnist_cnn import Net
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import Saliency


def onehot_encode(x, n_classes):
    result = torch.zeros(len(x), n_classes)
    result[range(len(x)), x] = 1
    return result


if __name__ == '__main__':
    model_path = "../models/saved_models/mnist_cnn.pt"
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=16, shuffle=True)

    images, labels = next(iter(train_loader))
    images.requires_grad = True

    logits = net(images)
    predictions = torch.argmax(logits, dim=1)
    ohe_pred = onehot_encode(predictions, 10)
    logits.backward(ohe_pred)
    sensitivities = torch.pow(images.grad, 2)

    image = images[0].detach().numpy()
    sensitivity = sensitivities[0].detach().numpy()

    plt.figure(0)
    plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
    plt.imshow(np.reshape(sensitivity, [28, 28]), cmap='hot', alpha=0.5)
    plt.colorbar()
    plt.title(f"Digit: {labels[0]} (using logits)")

    sal = Saliency(net)
    attr = sal.attribute(torch.tensor(image).unsqueeze(0), target=labels[0])
    attr = torch.pow(attr, 2)
    plt.figure(1)
    plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
    plt.imshow(np.reshape(attr, [28, 28]), cmap='hot', alpha=0.5)
    plt.colorbar()
    plt.title(f"Digit: {labels[0]} (using captum)")

"""
    images.grad.zero_()
    output = net(images)
    predictions = torch.argmax(net(images), dim=1)
    ohe_pred = onehot_encode(predictions, 10)
    output.backward(ohe_pred)
    sensitivities = torch.pow(images.grad, 2)

    image = images[0].detach().numpy()
    sensitivity = sensitivities[0].detach().numpy()

    plt.figure(1)
    plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
    plt.imshow(np.reshape(sensitivity, [28, 28]), cmap='hot', alpha=0.5)
    plt.colorbar()
    plt.title(f"Digit: {labels[0]} (using output)")
"""
"""
    plt.figure(figsize=(15, 15))
    for i in range(8):
        plt.subplot(8, 2, 2 * i + 1)
        plt.imshow(np.reshape(images[2 * i], [28, 28]), cmap='gray')
        plt.imshow(np.reshape(sensitivity[2 * i], [28, 28]), cmap='hot', alpha=0.5)
        plt.title('Digit: {}'.format(2 * i))
        plt.colorbar()

        plt.subplot(8, 2, 2 * i + 2)
        plt.imshow(np.reshape(images[2 * i + 1], [28, 28]), cmap='gray')
        plt.imshow(np.reshape(sensitivity[2 * i + 1], [28, 28]), cmap='hot', alpha=0.5)
        plt.title('Digit: {}'.format(2 * i + 1))
        plt.colorbar()

    plt.tight_layout()
"""
