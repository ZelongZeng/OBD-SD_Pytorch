import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import copy

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.margin             = opt.loss_margin_margin
        self.nu                 = opt.loss_margin_nu
        self.beta_constant      = opt.loss_margin_beta_constant
        self.beta_val           = opt.loss_margin_beta

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_margin_beta)

        self.batchminer = batchminer

        self.name  = 'margin'

        self.lr    = opt.loss_margin_beta_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ###
        # self.f_distillation_standard = opt.f_distillation_standard
        self.tau = opt.kd_tau
        self.alpha = opt.kd_alpha
        self.n_epochs = opt.n_epochs
        self.diffusion_w = opt.diffusion_w
        self.tau_diff = opt.tau_diff

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
        sampled_triplets = self.batchminer(batch, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss_rank = torch.sum(pos_loss+neg_loss)
            else:
                loss_rank = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: 
                beta_regularization_loss = torch.sum(beta)
                loss_rank += self.nu * beta_regularization_loss.to(torch.float).to(d_ap.device)
        else:
            loss_rank = torch.tensor(0.).to(torch.float).to(batch.device)

        # OBD-SD
        if self.PSD:
            similarity = batch.mm(batch.T)
            teacher_similarity = torch.mm(teacher_batch, teacher_batch.t())

            if self.OBDP:
                #### diffusion
                ##### standard symmetry normalization
                # masks = torch.eye(batch.size(0)).cuda()
                # affinity = copy.deepcopy(teacher_similarity)
                # affinity[range(batch.size(0)), range(batch.size(0))] = 0.
                # degree = torch.sum(affinity, dim=-1) + 1e-12
                # mat = (degree ** (-0.5)).repeat(batch.size(0), 1) * masks
                # S = mat @ affinity @ mat
                # W = (1 - self.diffusion_w) * torch.inverse(masks - self.diffusion_w * S)
                # target_cache_affinity = torch.matmul(W, teacher_similarity)
                # target_cache = self.softmax(target_cache_affinity, self.tau)

                ##### softmax-based normalization.
                ##### Here use softmax-based normalization is more convenient and achieve almost the same performance.
                masks = torch.eye(batch.size(0)).cuda()
                W = teacher_similarity - masks * 1e9
                W = self.softmax(W, self.diffusion_tau)
                W = (1 - self.diffusion_w) * torch.inverse(masks - self.diffusion_w * W)
                target_cache = torch.matmul(W, self.softmax(teacher_similarity, self.kd_tau))
            else:
                target_cache = self.softmax(teacher_similarity, self.tau)

            sim_cache = self.logsoftmax(similarity, self.kd_tau)
            loss_kd = self.kl_d(sim_cache, target_cache.detach())
            loss = torch.mean(loss_rank) + (epoch / self.n_epochs) * self.kd_lambda * (self.kd_tau ** 2) * loss_kd

        else:
            loss = torch.mean(loss_rank)

        return loss
