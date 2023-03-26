import torch, torch.nn as nn
import copy



"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.thresh     = opt.loss_multisimilarity_thresh

        self.name           = 'multisimilarity'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ###
        # self.f_distillation_standard = opt.f_distillation_standard
        self.kd_tau = opt.kd_tau
        self.kd_lambda = opt.kd_lambda
        self.n_epochs = opt.n_epochs
        self.diffusion_w = opt.diffusion_w
        self.diffusion_tau = opt.diffusion_tau

        self.PSD = opt.PSD
        self.OBDP = opt.OBDP

    def logsoftmax(self, x, tau):
        ls = torch.nn.LogSoftmax(dim=1)
        return ls(x/tau)

    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x/tau)

    def kl_d(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(input, target)

    def forward(self, batch, teacher_batch, labels, epoch, **kwargs):
        similarity = batch.mm(batch.T)

        loss_rank = []
        for i in range(len(batch)):
            pos_idxs       = labels==labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels!=labels[i]

            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            ### This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            loss_rank.append(pos_term + neg_term)

        loss_rank = torch.mean(torch.stack(loss_rank))

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
