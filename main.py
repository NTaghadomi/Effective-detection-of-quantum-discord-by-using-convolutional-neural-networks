# Seed all sources of randomness

from utils import random_state
import random
random.seed(random_state)
import numpy as np
np.random.seed(random_state)
import torch
torch.manual_seed(random_state)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

#%%

from torch.utils.data import DataLoader
from data_preparation_newmethod import X, Y
from models import Branching, CNN
from dataset import MyDataset
from initialize_weights import set_hermitian_weights
from training import train, test
from utils import print_cnn_weights, print_branching_paths
from utils import logging
from utils import plot, plot_branching, plot_conf_mat, plot_precision_recall
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import sys
import torch.nn.functional as F
from custom_kernels import set_custum_kernels


#%%
# set hyper parameters

params = {
    'batch_size': 512,
    'epochs': 20,
    'lr': 1e-3,
    # 'lr': 3e-4,
}


# %%
# Create Dataset and then DataLoader

dataset = MyDataset(X, Y)

dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)


# %%
# Create the model

net = Branching()

log = logging(net)

orig_stdout = sys.stdout
logfile = open(f'{log.path}/log.txt', 'w')
sys.stdout = logfile

if type(net) is Branching:
    
    acc, val, loss, acc_test_list = [], [], [], []
    recall_list, precision_list, f1_list = [], [], []
    # for b in range(1, 16):
    for b in [5, 8]:
        print(f"\nBranching {b}\n")
        net = Branching(b)
        
        # set_hermitian_weights(net)
        set_custum_kernels(net)

        acc_train, acc_val, loss_train = train(net, dataset, dataloader, epochs=params['epochs'], lr=params['lr'])
        acc_test = test(net, dataset)
        
        plot(acc_train, acc_val, loss_train, log.path, b)
        
        acc.append(acc_train[-1])
        val.append(acc_val[-1])
        loss.append(loss_train[-1])
        acc_test_list.append(acc_test)
        
        y_pred = net.round_sigmoid(net(dataset.X_test)).detach().numpy()
        cm = confusion_matrix(dataset.Y_test, y_pred)
        plot_conf_mat(cm, log.path, b)
        y_pred_pr = F.sigmoid(net(dataset.X_test)).detach().numpy()
        plot_precision_recall(dataset.Y_test, y_pred_pr, f"Branching - {b} Paths", log.path, b)
        
        print_branching_paths(net, log.path, b)
        
        precision, recall, f1score, _ = precision_recall_fscore_support(dataset.Y_test, y_pred, average='binary')
        print(f"Precision: {precision:.6f}   Recall: {recall:.6f}   f1score: {f1score:.6f}")
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1score)

    plot_branching(acc, val, acc_test_list, loss, log.path, precision_list, recall_list, f1_list)

else:
    acc_train, acc_val, loss_train = train(net, dataset, dataloader, epochs=params['epochs'], lr=params['lr'])
    acc_test = test(net, dataset)
    plot(acc_train, acc_val, loss_train, log.path)
    
    y_pred = net.round_sigmoid(net(dataset.X_test)).detach().numpy()
    cm = confusion_matrix(dataset.Y_test, y_pred)
    plot_conf_mat(cm, log.path)
    plot_precision_recall(dataset.Y_test, y_pred, "CNN", log.path)

    # Print conv weights
    print_cnn_weights(net, log.path)
    
    precision, recall, f1score = precision_recall_fscore_support(dataset.Y_test, y_pred, average='binary')
    print(f"Precision: {precision:.6f}   Recall: {recall:.6f}   f1score: {f1score:.6f}")



sys.stdout = orig_stdout
logfile.close()
