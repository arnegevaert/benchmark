from benchmark import models, datasets
import torch
from torch import optim, nn
from os import path
import argparse
from tqdm import tqdm


const_dict = {
    "resnet": models.Resnet,
    "densenet": models.Densenet,
    "vgg": models.Vgg,
    "squeezenet": models.Squeezenet,
    "alexnet": models.Alexnet,
    "mobilenet_v2": models.Mobilenet_v2
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--version", type=str, default="resnet18")
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {
        "output_logits": True,
        "num_classes": 10
    }
    if args.version:
        kwargs["version"] = args.version

    model = const_dict[args.model](**kwargs)
    model.to(device)
    dataset = datasets.Cifar(args.batch_size, path.join(args.data_root, "CIFAR10"))
    train_loader = dataset.get_dataloader(train=True)
    test_loader = dataset.get_dataloader(train=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        prog = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_samples, correct_samples = 0, 0
        losses = []
        for batch, labels in prog:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            y_pred = model(batch)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total_samples += batch.size(0)
            correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
            prog.set_postfix({"loss": sum(losses)/len(losses),
                              "acc": f"{correct_samples}/{total_samples} ({100*correct_samples/total_samples:.2f}%)"})
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_ep{epoch+1}.pt")
