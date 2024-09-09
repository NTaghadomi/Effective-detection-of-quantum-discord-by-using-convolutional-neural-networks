import torch
import torch.nn as nn
from qutip import rand_herm
import numpy as np

param_names = [
    "conv1.weight",
    "conv2.weight",
    "conv3.weight",
    "conv4.weight",
    "convx.weight",
    "convy.weight",
    "convz.weight",
    "convi.weight",
]


# def hermitian_uniform(size):
#     A = (
#         torch.rand(size, dtype=torch.cfloat) - 0.5
#     )  # Uniform distribution on [-0.5, 0.5)
#     return 0.5 * (A + A.conj().mT)


def set_weights(net, name):
    for n, p in net.named_parameters():
        if n == name:
            # print(p.data.shape)
            p.data = nn.parameter.Parameter(torch.tensor(np.array(rand_herm(p.data.shape[-1]), dtype=np.complex64).reshape(p.data.shape)))
            # p.data = nn.parameter.Parameter(hermitian_uniform(p.data.shape))
            print(f"Initialized {name} to Hermiti.")


def set_hermitian_weights(net):
    for name in param_names:
        set_weights(net, name)


def print_conv_weights(net):
    for n, p in net.named_parameters():
        if n in param_names:
            print(n, p)
