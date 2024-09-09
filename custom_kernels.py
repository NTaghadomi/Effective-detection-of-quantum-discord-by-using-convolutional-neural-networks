# -*- coding: utf-8 -*-

import numpy as np
import torch


def set_custum_kernels(net):
    if net.branch == 5:
        w1 = np.array([[0, 0, 1],
                       [.927, .19, -.334]])

        w2 = np.array([[.1066, .079, -.992],
                       [-.331, .767, .549],
                       [.5437, .1157, .81],
                       [-.077, .0253, .9967]])

    elif net.branch == 8:
        w1 = np.array([[-.025185, .194479, .98058],
                       [-.1444, -.9635, .2255]])

        w2 = np.array([[-.3918, -.7424, -.5435],
                       [-.7206, -.6653, .1952],
                       [.199, -.249, -.9478],
                       [.6869, -.7082, .163]])

    sigma_x = 1/2 * np.array([[0, 1],
                              [1, 0]])
    sigmas_y = 1/2 * np.array([[0, 1j],
                              [-1j, 0]])
    sigma_z = 1/2 * np.array([[1, 0],
                              [0, -1]])

    w1_kernels = ["conv1.weight", "conv2.weight"]
    w2_kernels = ["convx.weight", "convy.weight", "convz.weight", "convi.weight"]

    for i in range(2):
        n_x, n_y, n_z = w1[i, 0], w1[i, 1], w1[i, 2]
        k_i = sigma_x * n_x + sigmas_y * n_y + sigma_z * n_z
        for n, p in net.named_parameters():
            if n == w1_kernels[i]:
                p.data = torch.nn.parameter.Parameter(torch.tensor(k_i.astype(dtype=np.complex64).reshape(p.data.shape)))
                p.requires_grad = False
                print(f"Initialized {n} to k1_{i}. parameter froze.")
                break

    for i in range(4):
        n_x, n_y, n_z = w2[i, 0], w2[i, 1], w2[i, 2]
        k_i = sigma_x * n_x + sigmas_y * n_y + sigma_z * n_z
        for n, p in net.named_parameters():
            if n == w2_kernels[i]:
                p.data = torch.nn.parameter.Parameter(torch.tensor(k_i.astype(dtype=np.complex64).reshape(p.data.shape)))
                p.requires_grad = False
                print(f"Initialized {n} to k2_{i}. parameter froze.")
                break
