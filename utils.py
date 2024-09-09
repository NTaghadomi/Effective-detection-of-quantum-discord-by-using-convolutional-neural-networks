import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve


random_state = 741


def svd(data):
    U, S, Vh = torch.linalg.svd(data)
    s = torch.zeros((*S.shape, 4))
    for i in range(len(S)):
        for j in range(4):
            s[i][0] = torch.diag(S[i][0])
    data = torch.cat((U, s, Vh), dim=1)
    return data


# X_train = svd(X_train)
# X_val = svd(X_val)
# X_test = svd(X_test)


def print_cnn_weights(net, path):
    with open(f"{path}/weights.txt", "x") as file:
        for name, param in net.named_parameters():
            if name[:4] == "conv" and name[-6:] == "weight":
                print(name[:5] + ":\n", param.data, param.data.shape, "\n", file=file)


def print_branching_paths(net, path, branch):
    kernel = {}
    knames = ["conv1", "conv2", "conv3", "conv4", "convx", "convy", "convz", "convi"]
    for name, param in net.named_parameters():
        if name[:5] in knames:
            kernel[name[:5]] = param.data

    with open(f"{path}/weights_paths_branch{branch}.txt", "x") as file:
        print(f"Branch {branch}:\n", file=file)
        for kname, k in kernel.items():
            print(kname, k, "\n", file=file)


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return F.relu(z.real) + 1.0j * F.relu(z.imag)


class logging:
    def __init__(self, net, branch=None):
        self.root = "results"
        folders = list(os.walk(self.root))[1:]
        for folder in folders:
            if not folder[2]:
                os.rmdir(folder[0])

        date = str(datetime.now())[:19].replace(":", "-")
        net_type = re.findall(r"\.(.*?)\'", str(type(net)))[0]

        if branch:
            name = date + " " + net_type + " " + str(branch)
        else:
            name = date + " " + net_type

        print(f"Logging results in: './{self.root}/{name}/'")
        self.path = os.path.join(self.root, name)
        os.mkdir(self.path)


def plot(train, val, loss, path, branch=0):
    plt.figure()
    plt.plot(train, label="Training")
    plt.plot(val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{path}/Accuracy {branch}.png", dpi=300)

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)
    plt.savefig(f"{path}/Loss {branch}.png", dpi=300)


def plot_branching(train, val, test, loss, path, precision, recall, f1):
    plt.figure()
    x = np.arange(1, len(train)+1)
    plt.plot(x, train, label="Training", linestyle='--')
    plt.plot(x, val, label="Validation", linestyle='--')
    plt.plot(x, test, label="Test", linestyle='--')
    plt.xlabel("Branch")
    plt.ylabel("Accuracy")
    # plt.grid(True)
    plt.legend()
    plt.savefig(f"{path}/Accuracy.png", dpi=300)

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Branch")
    plt.ylabel("BCE Loss")
    # plt.grid(True)
    plt.savefig(f"{path}/Loss.png", dpi=300)

    plt.figure()
    plt.plot(x, precision, label="Precision", linestyle='--', marker="^")
    plt.plot(x, recall, label="Recall", linestyle='--', marker="D")
    plt.plot(x, f1, label="F1-score", linestyle='--', marker="X")
    plt.xlabel("Independent Convolution Path (m)")
    plt.ylabel("Precision & Recall & F1 Score")
    plt.legend()
    plt.savefig(f"{path}/PRF.png", dpi=300)

    plt.figure()
    plt.plot(x, test, label="Accuracy", linestyle='--', marker="^")
    plt.plot(x, recall, label="Recall", linestyle='--', marker="D")
    plt.plot(x, f1, label="F1-score", linestyle='--', marker="X")
    plt.xlabel("Independent Convolution Path (m)")
    plt.ylabel("Precision & Recall & F1 Score")
    plt.legend()
    plt.savefig(f"{path}/ARF.png", dpi=300)


def plot_conf_mat(cm, path, branch=None):
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    if branch:
        disp.figure_.savefig(f"{path}/ConfusionMatrix {branch}.png", dpi=300)
    else:
        disp.figure_.savefig(f"{path}/ConfusionMatrix.png", dpi=300)


def plot_precision_recall(y_true, y_pred, name, path, branch=None):
    plt.figure()
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(lr_recall, lr_precision, ls="--", linewidth=0.8, marker='d', markevery=10000)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    # disp = PrecisionRecallDisplay.from_predictions(
    #     y_true, y_pred, name=name, plot_chance_level=False, pos_label=1, drop_intermediate=True
    # )
    # disp.ax_.set_title("Precision-Recall curve")

    if branch:
        # disp.figure_.savefig(f"{path}/PrecisionRecall {branch}.png", dpi=300)
        plt.savefig(f"{path}/PrecisionRecall {branch}.png", dpi=300)
    else:
        # disp.figure_.savefig(f"{path}/PrecisionRecall.png", dpi=300)
        plt.savefig(f"{path}/PrecisionRecall.png", dpi=300)
