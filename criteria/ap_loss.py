import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from .smooth_ap_loss import SmoothAP

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.batchminer = batchminer
        self.name           = 'aploss'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        self.aploss = SmoothAP(0.01, opt.bs, int(opt.bs/opt.samples_per_class), opt.embed_dim)

        ###
        self.f_distillation_standard = opt.f_distillation_standard
        self.tau = opt.kd_tau
        self.alpha = opt.kd_alpha
        self.beta = opt.kd_beta
        self.n_epochs = opt.n_epochs

    def logsoftmax(self, x, tau):
        ls = torch.nn.LogSoftmax(dim=1)
        return ls(x / tau)

    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x / tau)

    def kl_d(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(input, target)

    def forward(self, batch, teacher_batch, labels, epoch, **kwargs):
        loss             = self.aploss(batch, labels)


        return loss
