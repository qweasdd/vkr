from datasets import VerificationDataset
from utils import coll, euclidean_dist, cosine_sim, CyclicLR
from models import VerificationModel
from losses import TripletLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import time
import math
import argparse

parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--csv-path', type=str, default='data/crop_train_reid.csv')
parser.add_argument('--data-path', type=str, default='data/verificaton_images/')
parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true')
parser.add_argument('--batch-size', type=int, default=16)
parser.set_defaults(use_cuda = False)
args = parser.parse_args()

gpu = args.use_cuda

train_data = VerificationDataset(args.csv_path, args.data_path)
trainloader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle= True, collate_fn= coll, drop_last= True, num_workers = 5)

num_classes = train_data.data['sdocid'].nunique()
model = VerificationModel(num_classes)
if gpu:
    model = model.to('cuda')
    
triplet_loss = TripletLoss(cuda = gpu)
optimizer = optim.Adam(model.parameters(), lr= 3.5e-4)

epochs  = 120
epoch_steps = len(train_data) / args.batch_size
steps = math.ceil(epoch_steps * 10)
first_scheduler = CyclicLR(optimizer, base_lr= 3.5e-5, max_lr=3.5e-4, step_size= steps)
second_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)

print('Training:\n')

for epoch in range(epochs):
    
    model.train()
    sum_loss = 0
    sum_tr_loss = 0
    sum_clf_loss = 0
    
    correct = 0
    overall = 0
    for n, (x, y) in enumerate(trainloader):
        
        classes = y.clone()
        if gpu:
            x = x.to('cuda')
            y = y.to('cuda')
        
        features, logits = model(x, y)
        
        dist = euclidean_dist(features, features)
        labels = torch.cuda.LongTensor(np.repeat(np.arange(dist.shape[0] // 3), 3)).unsqueeze(1)
        
        pos = labels.eq(labels.t())
        neg = labels.ne(labels.t())
        
        dist_ap = torch.max(dist[pos].view(dist.shape[0], -1), 1)[0]
        dist_an = torch.min(dist[neg].view(dist.shape[0], -1), 1)[0]
        
        optimizer.zero_grad()
        
        tr_loss = triplet_loss(dist_ap= dist_ap, dist_an = dist_an)
        clf_loss = F.cross_entropy(logits, y)
        loss = tr_loss + clf_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch < 10:
            first_scheduler.batch_step()
            
        sum_loss += loss.item()
        sum_tr_loss += tr_loss.item()
        sum_clf_loss += clf_loss.item()
        
        predict = logits.cpu().detach()
        
        correct += (predict.max(1)[1] == classes).sum().item()
        overall += labels.shape[0]
        
    if epoch >= 10:
        second_scheduler.step()
  
    print(time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
    print('Epoch : %d Train loss : %.5f Triplet : %.5f Clf: %.5f Accuracy:%.5f'%(epoch, sum_loss / (n + 1), sum_tr_loss / (n + 1), sum_clf_loss / (n + 1), correct / overall))
    print()
    
torch.save(model.state_duct(), 'verification_model')