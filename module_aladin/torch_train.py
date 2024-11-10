import os, natsort, re
from tqdm import tqdm
import time, random


from module_aladin.config import roles, parens, custom_hanja


from itertools import repeat, chain

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

import torch
from torch.utils.data import DataLoader
import math
import time
from torch import nn, optim
from torch.optim import Adam
import locale
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torcheval.metrics import functional as F_metric

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)


locale.getpreferredencoding = lambda: "UTF-8"


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    Y_actual, Y_pred = list(),list()
    for i, batch in enumerate(iterator):
        x,y_actual = batch[0], batch[1]
        optimizer.zero_grad()
        y_pred = model(x).reshape(-1)
        loss = criterion(y_pred,y_actual)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        Y_pred.append(y_pred)
        Y_actual.append(y_actual)

    Y_actual, Y_pred = torch.cat(Y_actual), torch.cat(Y_pred).reshape(-1)

    return epoch_loss / len(iterator), F_metric.r2_score(Y_pred,Y_actual).detach().cpu().numpy()


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    Y_actual, Y_pred = list(),list()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x,y_actual = batch[0], batch[1]
            y_pred = model(x).reshape(-1)
            loss = criterion(y_pred,y_actual)

            epoch_loss += loss.item()

            Y_pred.append(y_pred)
            Y_actual.append(y_actual)

    Y_actual, Y_pred = torch.cat(Y_actual), torch.cat(Y_pred).reshape(-1)

    return epoch_loss / len(iterator), F_metric.r2_score(Y_pred,Y_actual).detach().cpu().numpy()
  
def run(model,train_config,train_iter,valid_iter,total_epoch,warmup,best_loss,save_dir,expt_name):
    train_losses, valid_losses, train_scores, valid_scores = [], [], [], []
    optimizer, scheduler, criterion, clip = *train_config,
    best_epoch=0
    for step in range(total_epoch):

        train_loss, train_score = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, valid_score = evaluate(model, valid_iter, criterion)

        if step > warmup:
            scheduler.step(valid_loss)

        train_loss=math.sqrt(train_loss)
        valid_loss=math.sqrt(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_scores.append(train_score)
        valid_scores.append(valid_score)

        if valid_loss < best_loss:
            best_loss,best_epoch = valid_loss,step+1
            torch.save(model, os.path.join(save_dir,'best_{0}.pt'.format(expt_name)))

        print(f'Epoch: {step + 1:>5}\t\tlr: {scheduler.get_last_lr()[0]:.8f}')
        print(f'\tTrain Loss: {train_loss:.3f}\tTrain Score: {train_score:.3f}\tVal Loss: {valid_loss:.3f}\tVal Score: {valid_score:.3f}')

    print('Best Epoch: ',best_epoch)
    return model,train_losses,valid_losses,train_scores, valid_scores

def test(model, iterator, criterion,device,scatter=True,**kwargs):
    model.eval()
    epoch_loss = 0
    Y_actual, Y_pred = list(),list()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x,y_actual = batch[0].to(device), batch[1].to(device)
            x,y_actual = x.to(torch.float32),y_actual.to(torch.float32)
            y_pred = model(x).reshape(-1)
            loss = criterion(y_pred,y_actual)

            epoch_loss += loss.item()

            score_list = list()
            Y_pred.append(y_pred)
            Y_actual.append(y_actual)

    Y_actual, Y_pred = torch.cat(Y_actual), torch.cat(Y_pred).reshape(-1)
    Y_rslt,Y_truth=Y_pred.detach().cpu().numpy(),Y_actual.detach().cpu().numpy()
    if scatter:
      fig,ax = plt.subplots()
      sns.scatterplot(x=Y_truth,y=Y_rslt)
    else :
      g = sns.jointplot(x=Y_truth, y=Y_rslt, kind="hex", xlim = (0,60000), ylim = (0,60000),**kwargs)
    score = r2_score(Y_truth,Y_rslt)
    mape = mean_absolute_percentage_error(Y_truth,Y_rslt)
    print(math.sqrt(epoch_loss / len(iterator)), "\tr2 : ",score,"\tmape : ",mape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
    
def trainer_setting(model,init_lr,weight_decay,adam_eps,factor,patience):
  print(f'The model has {count_parameters(model):,} trainable parameters')
  model.apply(initialize_weights)
  optimizer = Adam(params=model.parameters(),
                   lr=init_lr,
                   weight_decay=weight_decay,
                   eps=adam_eps)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                   verbose=True,
                                                   factor=factor,
                                                   patience=patience)

  criterion = nn.MSELoss()
  return(model,optimizer,scheduler,criterion)