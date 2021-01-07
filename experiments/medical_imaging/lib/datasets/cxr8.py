from __future__ import division

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from PIL import Image




class Cxr8(Dataset):
    def __init__(self, data_location, train, toy=False, imsize=512,train_weighted=True,valid_weighted=True):
        super(Cxr8, self).__init__()

        data_split = "train" if train else "valid"
        df = pd.read_csv(os.path.join(data_location,"%s_relabeled.csv" % (data_split)))

        if toy:
            df = df.sample(frac=0.01)

        self.df = df
        self.img_paths = [os.path.join(data_location, pth) for pth in df["Path"].tolist()]
        self.pathologies = [col for col in df.columns.values if col != "Path"]

        self.labels = df[self.pathologies].to_numpy().astype(int)

        self.n_classes = self.labels.shape[1]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if data_split == "train":
            transforms_lst = [
                transforms.Resize((imsize, imsize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
            self.transform = transforms.Compose([t for t in transforms_lst if t])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                normalize,
            ])
        self.df = df


        if (data_split == "train" and train_weighted) or (data_split == "valid" and valid_weighted):
            self.get_weights(data_split)


    def get_weights(self,  data_split, device="cpu"):

        self.use_gpu = torch.cuda.is_available() and device !="cpu"
        p_count = (self.labels == 1).sum(axis = 0)
        self.p_count = p_count
        n_count = (self.labels == 0).sum(axis = 0)
        total = p_count + n_count

        # invert *opposite* weights to obtain weighted loss
        # (positives weighted higher, all weights same across batches, and p_weight + n_weight == 1)
        p_weight = n_count / total
        n_weight = p_count / total

        self.p_weight_loss = Variable(torch.FloatTensor(p_weight), requires_grad=False)
        self.n_weight_loss = Variable(torch.FloatTensor(n_weight), requires_grad=False)

        print ("Positive %s Loss weight:" % data_split, self.p_weight_loss.data.numpy())
        print ("Negative %s Loss weight:" % data_split, self.n_weight_loss.data.numpy())
        random_loss = sum((p_weight[i] * p_count[i] + n_weight[i] * n_count[i]) *\
                                               -np.log(0.5) / total[i] for i in range(self.n_classes)) / self.n_classes
        print ("Random %s Loss:" % data_split, random_loss)


    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        label = self.labels[index]

        return self.transform(img), torch.FloatTensor(label)

    def __len__(self):
        return len(self.img_paths)

    def weighted_loss(self, preds, target, epoch=1):

        weights = target.type(torch.FloatTensor) * (self.p_weight_loss.expand_as(target)) + \
                  (target == 0).type(torch.FloatTensor) * (self.n_weight_loss.expand_as(target))
        if self.use_gpu:
            weights = weights.cuda()
        loss = 0.0
        for i in range(self.n_classes):
            loss += nn.functional.binary_cross_entropy_with_logits(preds[:,i], target[:,i], weight=weights[:,i])
        return loss / self.n_classes



def loader_to_gts(data_loader):
    gts = []
    for (inputs, labels) in data_loader:
        for label in labels.cpu().numpy().tolist():
            gts.append(label)
    gts = np.array(gts)
    return gts


def load_data(args):

    train_dataset = Dataset(args, "train")
    valid_dataset = Dataset(args, "valid")

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)

    return train_loader, valid_loader
