import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
from experiments.imagenette_downsizing.models import Resnet18, Resnet50
import os
import numpy as np
import argparse
from collections import deque
import time
import sklearn.metrics as metrics


device ='cuda'
pre_train = True

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("-s", "--im_size", type=int, default=224)
    parser.add_argument("-b","--batch_size", type=int, default=32)
    parser.add_argument( "-e","--epochs", type=int, default=10)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0002)
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--param_loc", type=str, default=None)#not needed
    parser.add_argument("--out", type=str, default=None)

    #--model_name resnet18 -s 64 -e 10 --target 0.4 -lr 1e-3 --data_loc D:\Project\Benchmark_branch_axel\data\imagenette2 --out outputs_scaling_loss_imagenette

    args = parser.parse_args()
    return args


def calculate_metrics(model,crit,dl):
    gt=[]
    preds=[]
    losses = []
    model.eval()
    with torch.no_grad():
        for batch, labels in dl:
            batch, labels = batch.to(device,non_blocking=True), labels.type(torch.long).to(device,non_blocking=True)
            out = model(batch)
            loss = crit(out, labels)
            losses.append(loss.item())
            out = torch.softmax(out,1).detach().cpu().numpy()
            preds.append(out)
            gt.extend(labels.detach().cpu().numpy())
    preds = np.vstack(preds)
    gt = np.array(gt)
    val_loss = np.mean(losses)
    Sens = metrics.recall_score(gt,preds.argmax(axis=1),average='macro')
    acc = metrics.accuracy_score(gt,preds.argmax(axis=1))
    print('avg_Sens: {:.4f} acc: {:.4f} Loss: {:.4f}'.format(Sens,acc,val_loss))
    return Sens,acc,val_loss


def train(model, crit, opt, epochs, train_dl, val_dl, schedule, early_stopping):
    for e in range(epochs):
        print("Epoch {}/{}".format(e, epochs))
        print("-" * 15)
        tic()
        model.train()
        train_losses = []
        for batch, labels in train_dl:
            opt.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            loss = crit(model(batch),labels)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            schedule.step()
        print("Training Loss: {:.6f}".format(np.mean(train_losses)))

        gt = []
        preds = []
        losses = []
        model.eval()
        with torch.no_grad():
            for batch, labels in val_dl:
                batch, labels = batch.to(device,non_blocking=True), labels.to(device,non_blocking=True)
                out = model(batch)
                loss = crit(out, labels)
                losses.append(loss.item())
                out = torch.softmax(out,1).detach().cpu().numpy()
                preds.extend(out)
                gt.extend(labels.detach().cpu().numpy())
            gt = np.array(gt)
            preds= np.array(preds)
            val_loss = np.mean(losses)
            Sens = metrics.recall_score(gt, preds.argmax(axis=1), average='macro')
            acc = metrics.accuracy_score(gt, preds.argmax(axis=1))
            print('avg_Sens: {:.4f} acc: {:.4f} Loss: {:.4f}'.format(Sens, acc, val_loss))
        toc()
        # schedule.step(val_loss)
        early_stopping.step(val_loss,model)
        if early_stopping():
            break
    return val_loss, Sens, acc



def get_data(im_size, batch_size, data_loc):
    transform_train = transforms.Compose([
    transforms.Resize((im_size,im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_loc,'train'), transform=transform_train)

    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    transform_val = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_loc,'val'), transform=transform_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl

class EarlyStopping():
    # stop when value has reached target
    def __init__(self, target_acc, eps=0., increase=True):
        self.target = target_acc
        self.stop = False
        self.inc =increase
        self.eps = eps

    def __call__(self):
        return self.stop
    def step(self,value, model):
        # if close enough, start saving models
        save = self.target -self.eps <= value if self.inc else self.target + self.eps >= value
        self.stop = self.target <= value if self.inc else self.target >= value
        if save and not self.stop:
            torch.save(model.state_dict,"outputs_scaling_loss_imagenette\\tmp\\tmp_model_value_{}".format(value))



if __name__ == '__main__':
    args = get_args()
    if args.model_name == 'resnet18':
        model = Resnet18(10,pretrained=pre_train,params_loc=args.param_loc)
    elif args.model_name == 'resnet50':
        model = Resnet50(10,pretrained=pre_train,params_loc=args.param_loc)
    else:
        raise Exception
    model.to('cuda')
    train_dl, val_dl = get_data(args.im_size, args.batch_size, args.data_loc)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    # schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1,cooldown=0, threshold=0.001, factor=0.5, verbose=True)
    schedule = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs, steps_per_epoch=len(train_dl))
    early_stop = EarlyStopping(args.target,eps=0.01,increase=False)
    train(model, criterion, optimizer, args.epochs, train_dl, val_dl, schedule, early_stop)

    Sens, acc, val_loss = calculate_metrics(model,criterion,val_dl)
    print(Sens, acc, val_loss)
    save_loc=os.path.join(args.out,
                            args.model_name + '_{}_size_{}_val{:.3f}_acc{:.3f}.pt'.format("imagenette",args.im_size, val_loss, acc))
    torch.save(model.state_dict(), save_loc)