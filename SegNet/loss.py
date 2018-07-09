#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxWeightedLoss(nn.Module):
    def __init__(self, epsilon = 1e-10, cuda=True):
        super(SoftmaxWeightedLoss, self).__init__()
        self.weights = torch.tensor([
            0.2595,
            0.1826,
            4.5640,
            0.1417,
            0.9051,
            0.3826,
            9.6446,
            1.8418,
            0.6823,
            6.2478,
            7.3614
            ])
        if cuda:
            self.weights = self.weights.cuda()

    def forward(self, logits, labels):
        softmax = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 11)
        #  return softmax
        return F.cross_entropy(softmax, labels, weight = self.weights)



if __name__ == "__main__":
    LossOp = SoftmaxWeightedLoss().cuda()
    in_tensor = torch.randn(4, 11, 1024, 1024).cuda()
    import numpy as np
    label = torch.tensor(np.ones((4, 1024, 1024), dtype=np.int64)).cuda()
    out = LossOp(in_tensor, label)
    device = torch.device("cpu")
    out = out.to(device)

    in_tensor = torch.tensor(np.ones((4, 11, 1024, 1024))).cuda()
    out = LossOp(in_tensor, label).to(device)
    print(out.shape)
    print(out[1,:])

