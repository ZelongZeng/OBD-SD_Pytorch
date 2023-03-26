"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import faiss, matplotlib.pyplot as plt, os, numpy as np, torch
from PIL import Image

import parameters    as par


"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.noisy_label_parameters(parser)
parser = par.load_checkpoint(parser)
parser = par.LSD_training_parameters(parser)

##### Read in parameters
opt = parser.parse_args()

# opt.evaluation_metrics = ['mAP_lim_visualization']


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
if opt.savename=='group_plus_seed':
    if opt.log_online:
        opt.savename = opt.group+'_s{}'.format(opt.seed)
    else:
        opt.savename = ''

"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import metrics       as metrics

#######################
def evaluate(metric_computer, dataloaders, model, opt, evaltypes, device):
    """
    Parent-Function to compute evaluation metrics, print summary string and store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos = metric_computer.compute_standard(opt, model, dataloaders[0], evaltypes, device)

    return computed_metrics, extra_infos

##########################
"""==================================================================================================="""
full_training_start_time = time.time()
"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'
opt.pretrained = not opt.not_pretrained

"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
# if not opt.use_data_parallel:
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])

"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)

"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
model_Recall      = archs.select(opt.arch, opt)
_  = model_Recall.to(opt.device)
model_NMI      = archs.select(opt.arch, opt)
_  = model_NMI.to(opt.device)

#################### LOAD CHECKPOINT ###################
if opt.PSD:
    if opt.OBDP:
        if opt.embed_dim == 128:
            model_Recall.load_state_dict(torch.load('./Training_Results/Recall/OBDSD_d128/'
                                                    + opt.dataset + '/net.pth'))

            model_NMI.load_state_dict(torch.load('./Training_Results/NMI/OBDSD_d128/'
                                                 + opt.dataset + '/net.pth'))
        else:
            model_Recall.load_state_dict(torch.load('./Training_Results/Recall/OBDSD_d512/'
                                                    + opt.dataset + '/net.pth'))

            model_NMI.load_state_dict(torch.load('./Training_Results/NMI/OBDSD_d512/'
                                                 + opt.dataset + '/net.pth'))
    else:
        model_Recall.load_state_dict(torch.load('./Training_Results/Recall/PSD/'
                                                  + opt.dataset + '/net.pth'))

        model_NMI.load_state_dict(torch.load('./Training_Results/NMI/PSD/'
                                                  + opt.dataset + '/net.pth'))




"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets    = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
if opt.use_tv_split:
    dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)
train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)

opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)
"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)

"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}
epoch_start_time = time.time()

"""======================================="""
### Evaluate Metric for Training & Test (& Validation)
_ = model_Recall.eval()
_ = model_NMI.eval()

print('\nComputing Testing Metrics for model_Recall...')
mat_Recall, extra_Recall = evaluate(metric_computer, [dataloaders['testing']],
                                        model_Recall, opt, opt.evaltypes, opt.device)

print('\nComputing Testing Metrics for model_NMI...')
mat_NMI, extra_NMI = evaluate(metric_computer, [dataloaders['testing']],
                                        model_NMI, opt, opt.evaltypes, opt.device)

print('Recall@1: ', mat_Recall['discriminative']['e_recall@1'])
print('NMI: ', mat_NMI['discriminative']['nmi'])



print('Total Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
print('\n-----\n')



