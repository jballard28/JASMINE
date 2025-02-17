# Functions for computing contrastive loss
# - This function computes multi-way contrastive loss, where there are multiple positive and negative samples. Based on soft-nearest-neighbors loss (https://lilianweng.github.io/posts/2021-05-31-contrastive/)

import torch
from torch.nn import CosineSimilarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np

def contrastive_loss(usamp, pmsk, net, temperature):

    # usamp: map (modality name, embeddings), where embeddings is (batch_size x embedding_dim)
    # pmsk: mask denoting which modalities are available for that sample (batch_size x n_modalities)

    # positive pairs: different modalities from the same sample

    n_mod = pmsk.shape[1]
    batch_size = pmsk.shape[0]
    latent_dim = usamp[net.keys[0]].shape[1]

    usamp_cat = torch.cat([usamp[k] for k in net.keys],dim=0)
    pmsk_flat = pmsk.t().flatten()
    cos_mat = pairwise_cosine_similarity(usamp_cat)

    maskmat = torch.zeros(cos_mat.shape,device=net.device)
    for i in range(batch_size):
        maskmat[i*n_mod:(i+1)*n_mod,i*n_mod:(i+1)*n_mod] = 1
    maskmat = maskmat - torch.eye(cos_mat.shape[0],device=net.device)
    allbutself = 1 - torch.eye(cos_mat.shape[0],device=net.device)

    transf_mat = torch.exp(-cos_mat/temperature)
    numer = torch.sum(torch.mul(transf_mat,maskmat),dim=1)
    denom = torch.sum(torch.mul(transf_mat,allbutself),dim=1)

    loss = -torch.mean(torch.log(torch.div(numer,denom)))
    

    return loss

def perturb_samp(x, xrep, net, perturb_prop):

    if perturb_prop is None:
        perturb_prop = 0.25

    x_pert = {}
    xrep_pert = {}
    all_x = {}
    all_xrep = {}

    for k in net.keys:
        temp = x[k].clone()

        # effectively resampling each column (with replacement)
        rand_idcs = torch.randint(temp.shape[0],temp.shape, device=net.device)
        samp_vals = torch.take_along_dim(temp,rand_idcs,dim=0)
        # determining which values to perturb
        perturb_mask = torch.from_numpy(np.random.choice([0,1], size=temp.shape, p=[perturb_prop,1-perturb_prop])).to(net.device)
        # perturbing values by replacing them with resampled values
        x_pert[k] = torch.mul(temp,perturb_mask) + torch.mul(samp_vals,1-perturb_mask)
        xrep_pert[k] = x_pert[k]

#        # Generating new xrep (and standardizing x_pert) - same as what was done on original data
#        temp1 = x_pert[k]
#        mean = torch.mean(temp1, axis=0)
#        std = torch.std(temp1, axis=0)
#        x_pert[k] = (temp1 - mean) / std
#        xrep_pert[k] = (temp1 - mean) / std

        # concatenating x and x_pert for faster processing
        all_x[k] = torch.cat((x[k],x_pert[k]),dim=0)
        all_xrep[k] = torch.cat((xrep[k],xrep_pert[k]),dim=0)


    return all_x, all_xrep

def contrast_samp(jointsamp, net, temperature):
    # jointsamp: 2*batch_size x embedding_dim - first half are original data, second half are augmented data

    batch_size = int(jointsamp.shape[0]/2)

    # Get pairwise cosine loss across all samples, including augmented
    cos_mat = pairwise_cosine_similarity(jointsamp)

    maskmat = torch.zeros((batch_size,cos_mat.shape[1]),device=net.device)
    for i in range(batch_size):
        maskmat[i,i] = 1
        maskmat[i,i+batch_size] = 1
    maskmat = torch.cat((maskmat,maskmat), dim=0)
    maskmat = maskmat - torch.eye(cos_mat.shape[0],device=net.device)
    allbutself = 1 - torch.eye(cos_mat.shape[0],device=net.device)

    transf_mat = torch.exp(cos_mat/temperature)
    numer = torch.sum(torch.mul(transf_mat,maskmat),dim=1)
    denom = torch.sum(torch.mul(transf_mat,allbutself),dim=1)

    loss = -torch.mean(torch.log(torch.div(numer,denom)))

    return loss
