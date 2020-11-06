import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models
from experiments.lib.models import Alexnet, Resnet
from experiments.lib.datasets import CBIS_DDSM_patches
import os
import numpy as np
import argparse
from collections import deque
import time
import sklearn.metrics as metrics

start_time = deque()
def tic():
    start_time.append(time.time())
def toc():
    try:
        st = start_time.pop()
        t = time.time() - st
        print("elapsed time: {}".format(time.strftime("%H:%M:%S",time.gmtime(t))))
        return t
    except:
        return None

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--param_loc',type=str)
    parser.add_argument('--data_loc', required=True, type=str)
    parser.add_argument('-b','--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--validate', action="store_true")
    return parser

def train_epoch(net, opt,crit,dl):
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses=[]
    for batch,labels in dl:
        batch, labels = batch.cuda(), labels.type(torch.long).cuda()
        opt.zero_grad()
        out = net(batch)
        loss = crit(out,labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_loss = np.mean(losses)
    print("Training Loss: {:.6f}".format(train_loss))
    return train_loss

def validate_epoch(net,crit,dl):
    losses = []
    with torch.no_grad():
        for batch, labels in dl:
            batch, labels = batch.cuda(), labels.type(torch.long).cuda()
            out = net(batch)
            loss = crit(out, labels)
            losses.append(loss.item())
    val_loss = np.mean(losses)
    print("Validation Loss: {:.6f}".format(val_loss))
    return val_loss

def calculate_metrics(model,crit,dl):
    gt=[]
    preds=[]
    losses = []
    with torch.no_grad():
        for batch, labels in dl:
            batch, labels = batch.cuda(), labels.type(torch.long).cuda()
            out = model(batch)
            loss = crit(out, labels)
            losses.append(loss.item())
            out = torch.softmax(out,1).detach().cpu().numpy()
            preds.append(out)
            gt.extend(labels.detach().cpu().numpy())
    preds = np.vstack(preds)
    gt = np.array(gt)
    val_loss = np.mean(losses)
    acc = metrics.accuracy_score(gt,preds.argmax(axis=1))
    AUC = metrics.roc_auc_score(gt,preds[:,1])
    print('ROC AUC: {:.4f} Acc: {:.3f} Loss: {:.4f}'.format(AUC,acc,val_loss))





def run(args):
    train_ds = CBIS_DDSM_patches(args.data_loc,train=True,imsize=224)
    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_ds = CBIS_DDSM_patches(args.data_loc, train=False, imsize=224)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if args.model == 'Alexnet':
        model = Alexnet(2, params_loc=args.param_loc)
    elif args.model == 'Resnet':
        model = Resnet('resnet50',2,params_loc=args.param_loc)
    else:
        raise Exception("{} is not a valid model.".format(args.model))
    model.cuda()

    # for p in model.features.parameters():
    #     p.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    if not args.validate:
        train_loop(args, criterion, model, train_dl, val_dl)
    calculate_metrics(model,criterion,val_dl)


def train_loop(args, criterion, model, train_dl, val_dl):
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,nesterov=True)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.1, verbose=True)
    best_weights, best_loss = model.state_dict(), float("inf")
    counter = 0
    for e in range(args.epochs):
        tic()
        print("Epoch {}/{}".format(e, args.epochs))
        print("-" * 10)
        train_loss = train_epoch(model, optimizer, criterion, train_dl)
        val_loss = validate_epoch(model, criterion, val_dl)
        toc()
        schedule.step(val_loss)

        if val_loss < best_loss:
            best_weights = model.state_dict()
            best_loss = val_loss
            counter = 0
            print("Model improved")
        else:
            counter += 1
        if counter > 5: break
    torch.save(best_weights, args.model + '.pt')
    print("Best Validation Loss:", best_loss)
    model.load_state_dict(best_weights)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.data_loc)
    run(args)