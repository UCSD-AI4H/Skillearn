from isic_dataset import MyDataset, MyDataset2, trainDataset, validDataset, testDataset

import argparse
import os, time, glob
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display
import torchvision.datasets as dset
from torch.utils.data import DataLoader, ConcatDataset, random_split
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import sys
import copy
import math
import unittest
import pickle
from skimage import io, transform
from torchvision import transforms

sys.path.insert(0, "./../..")
from data import *
from utils import *
from grouping import MoE








parser = argparse.ArgumentParser("LBG")
parser.add_argument( "--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--batchsz", type=int, default=16, help="batch size")
parser.add_argument("--warmup", type=int, default=0, help="num of training warmup epochs")
parser.add_argument('--dataset', type=str, default='isic18', help='[isic18, cifar10]')
parser.add_argument("--lr", type=float, default=5e-4, help="init learning rate")
parser.add_argument("--lr_min", type=float, default=0.0, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--wd", type=float, default=3e-4, help="weight decay")
parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")
parser.add_argument("--cat_lr", type=float, default=6e-5, help="learning rate for cat encoding")
parser.add_argument("--cat_wd", type=float, default=1e-3, help="weight decay for cat encoding")
parser.add_argument("--cat_steps", type=int, default=5, help="cat steps")
parser.add_argument("--init_ch", type=int, default=16, help="num of init channels")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_len", type=int, default=16, help="cutout length")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
parser.add_argument("--train_portion", type=float, default=1, help="portion of training/val splitting")
parser.add_argument("--unroll_steps", type=int, default=5, help="unrolling steps")
parser.add_argument("--report_freq", type=int, default=100, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--lam", type=float, help="lambda", default=1)
parser.add_argument('--num_experts', type=int, default=2, help='number of experts')
parser.add_argument("--gamma", type=float, help="gamma", default=1)
parser.add_argument('--k', type=int, default=1)
parser.add_argument("--load_balance", action="store_true", default=False, help="use load balance")
parser.add_argument('--image_size', type=int, default=224, help="image size: 32, 224, 299")
parser.add_argument('--classes', type=int, default=7, help="number of classes")
# parser.add_argument('--set', type=str, default='test', help='experiment name')
parser.add_argument("--fair", action="store_true", default=False, help="fair train")




args = parser.parse_args()
print(args)

def save_checkpoint(state, checkpoint=args.save, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, f'log_lbc_{args.dataset}.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")

inp_size = args.image_size

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

if args.dataset == "isic18":
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    if args.image_size == 299:
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=360, shear=15),
            transforms.Resize(size=374),
            transforms.CenterCrop(size=374),
            transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0), ratio=(1.0, 1.01)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(size=374),
            transforms.CenterCrop(size=336),
            transforms.Resize(size=299),
            # transforms.ColorConstancy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif args.image_size == 224:
        train_transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.ToPILImage(),
                    # transforms.RandomResizedCrop(size=360, scale=(0.8, 1.0)),
                    transforms.Resize(256),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=args.image_size),  # Image net standards
                    # transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                    ])
        valid_transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.ToPILImage(),
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=args.image_size),
                    # transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])



    weight_per_class = [0.] * args.classes
    df = pd.read_csv('ISIC2018_Task3_Training_GroundTruth.csv')
    # print(len(df))
    display(df.head(2))                  
    count = []
    for cl in classes:
        val = int(df[cl].sum())
        count.append(val)
    # print(count)


    N = float(sum(count))
    for i in range(args.classes):                                                   
        weight_per_class[i] = N/float(count[i])  
    # print(weight_per_class)

    if os.path.exists('./data/ISIC18_train.pkl'):
        with open('./data/ISIC18_train.pkl', 'rb') as f:
            ISIC18_train = pickle.load(f)

    weights = [0] * len(ISIC18_train)                                              
    for idx, val in enumerate(ISIC18_train):                                          
        weights[idx] = weight_per_class[val[1]]  
    weights = torch.DoubleTensor(weights)                                       
    # print(weights)

    filenames = [X[0] for X in ISIC18_train]

    train_data = trainDataset(filenames, transform=train_transform)
    
    if os.path.exists('./data/ISIC18_test.pkl'):
        with open('./data/ISIC18_test.pkl', 'rb') as f:
            ISIC18_test = pickle.load(f)
    filenames2 = [X2[0] for X2 in ISIC18_test]
    valid_data = validDataset(filenames2, transform=train_transform)

    
    num_train = len(train_data)  
    # print(num_train)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    warmup = args.warmup

    train_queue = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=args.batchsz,
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(indices[:split])) ,
                    pin_memory=True,
                    num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=args.batchsz,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2)


    report_freq = int(num_train * args.train_portion // args.batchsz + 1)

    train_iters = int(args.epochs* (num_train * args.train_portion // args.batchsz + 1)* args.unroll_steps)




class CategoriesModule(nn.Module):
    def __init__(self, n=3072, k=3):
        super(CategoriesModule, self).__init__()
        self.C_mat = nn.Parameter(torch.randn(n, k))
        
        with torch.no_grad():
            # initialize to smaller value
            self.C_mat.mul_(1e-3)
            
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        torch.save(self.C_mat, 'categories_GAM.pt')
        return x @ self.C_mat


class Outer(ImplicitProblem):
    def forward(self, x):
        return self.module(x)
    
    def training_step(self, batch):
        if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "celeba":
            x, target = batch
        elif args.dataset == "isic18":
            x, target,_ = batch
        x, target = x.to(device), target.to(device)
        
        
        train = False
        Cats = self.forward(x.float())
        loss, correct = self.inner.module.loss(x, Cats, train, target, acc=True,aux=args.load_balance)
        acc = correct/x.size(0)
        assert not math.isnan(loss)
        epoch = int(self.count*(args.batchsz+1)*args.unroll_steps//(num_train * args.train_portion))

        return {"loss": loss, "acc": acc}
        

    def configure_train_data_loader(self):
        return valid_queue


    def configure_module(self):
        return CategoriesModule(n=3*args.image_size*args.image_size, k=args.num_experts).to(device)

    def configure_optimizer(self):
        optimizer = optim.Adam(
            self.module.parameters(),
            lr=args.cat_lr,
            betas=(0.5, 0.999),
            weight_decay=args.cat_wd,
        )
        return optimizer


class Inner(ImplicitProblem):
    def forward(self, x, Cats, train):
        return self.module(x, Cats, train)
    
    def training_step(self, batch):
        if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "celeba":
            x, target = batch
            x, target = x.to(device), target.to(device)
        elif args.dataset == "isic18":
            x, target, sf = batch
            x, target, sf = x.to(device), target.to(device), sf.to(device)
        train = True
        Cats = self.outer(x.float())
        
        if args.fair and args.dataset == "isic18":
            loss, correct = self.module.loss(x, Cats, train, target, acc=True, aux=False, sensitive_features=sf)
        else:
            loss, correct = self.module.loss(x, Cats, train, target, acc=True, aux=False, sensitive_features=None)
        acc = correct/x.size(0)
        return loss


    def configure_train_data_loader(self):
        return train_queue
        

    def configure_module(self):
        return MoE(
                   input_size=inp_size*inp_size*3,
                   output_size= args.classes, 
                   num_experts=args.num_experts, 
                   batch_size=args.batchsz, 
                   args=args,
                   noisy_gating=False, 
                   k=args.k, 
                   device=device).to(device)



    def configure_optimizer(self):
        optimizer = optim.SGD(
            self.module.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(train_iters // args.unroll_steps), eta_min=args.lr_min)
#         scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5000, 7500], gamma=0.1)
        return scheduler


best_acc = -1


class CategoryEngine(Engine):
    @torch.no_grad()
    def validation(self):
        corrects = 0
        total = 0
        global best_acc
        train = False
        objs = AvgrageMeter()
        if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "celeba":
            for x, target in valid_queue:
                x, target = x.to(device), target.to(device)
                n = x.size(0)
                Cats = self.outer(x.float())
                with torch.no_grad():
                    # out = self.inner(x, Cats, train)
                    out,_ = self.inner(x, Cats, train)
                    loss, correct = self.inner.module.loss(x, Cats, train, target, acc=True,aux=False)
                corrects += (out.argmax(dim=1) == target).sum().item()
                # loss, correct = self.inner.module.loss(x, Cats, train, target, acc=True)
                total += x.size(0)
                objs.update(loss.item(), n)
                # corrects += correct
            acc = corrects / total * 100
            print("[*] Valid Acc.:", acc)
            # torch.save(CategoriesModule.C_mat, 'categories_GAM.pt')
            if best_acc < acc:
                # torch.save(CategoriesModule.C_mat, 'categories_GAM_best.pt')
                best_acc = acc
                print("[*] Best Acc.:", best_acc)
                save_checkpoint(state=self.inner.module.state_dict(), filename='checkpoint_cifar.pth.tar')
            logging.info('[*] Acc.: %f || Best Acc.: %f || Loss: %f', acc, best_acc, objs.avg)
            return {"acc": acc, "best_acc": best_acc}
        
        elif args.dataset == "isic18":
            for x, target,_ in valid_queue:
            # for x, target,_ in valid_queue:
                x, target = x.to(device), target.to(device)
                n = x.size(0)
                Cats = self.outer(x.float())
                with torch.no_grad():
                    # out = self.inner(x, Cats, train)
                    out,_ = self.inner(x, Cats, train)
                    loss, correct = self.inner.module.loss(x, Cats, train, target, acc=True,aux=False)
                corrects += (out.argmax(dim=1) == target).sum().item()
                # loss, correct = self.inner.module.loss(x, Cats, train, target, acc=True)
                total += x.size(0)
                objs.update(loss.item(), n)
                # corrects += correct
            acc = corrects / total * 100
            print("[*] Valid Acc.:", acc)
            save_checkpoint(state=self.inner.module.state_dict(), filename='checkpoint_isic.pth.tar')
            if best_acc < acc:
                # torch.save(CategoriesModule.C_mat, 'categories_GAM_best.pt')
                best_acc = acc
                print("[*] Best Acc.:", best_acc)
                save_checkpoint(state=self.inner.module.state_dict(), filename='checkpoint_isic_best.pth.tar')
            logging.info('[*] Valid Acc.: %f || Best Acc.: %f || Loss: %f', acc, best_acc, objs.avg)
            return {"acc": acc, "best_acc": best_acc}




outer_config = Config(retain_graph=True, log_step=1)
inner_config = Config(type="darts", unroll_steps=args.unroll_steps, warmup_steps=args.warmup)
# inner_config = Config(type="darts", unroll_steps=args.unroll_steps, warmup_steps=args.warmup,allow_unused=True, log_step=1)
engine_config = EngineConfig(
    valid_step=args.report_freq,
    train_iters=train_iters,
    roll_back=True,
)

outer = Outer(name="outer", config=outer_config, device=device)
inner = Inner(name="inner", config=inner_config, device=device)

problems = [outer, inner]
u2l = {outer: [inner]}
l2u = {inner: [outer]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = CategoryEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()

