import os
import sys

import torch
import torchvision
import anndata as ad
import networkx as nx
import scanpy as sc
import sys
sys.path.insert(0,'../model')
import jasmine
import jasmine.models
from matplotlib import rcParams

import pickle
import numpy as np
import pandas as pd
import yaml
import argparse
import random
import math

def main():

    parser = argparse.ArgumentParser(description='train JASMINE model on ADNI multiomics')
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('-k', '--lam_kl', type=float, default=0.97, required=False)
    parser.add_argument('-c', '--lam_cos', type=float, default=3.0, required=False)
    parser.add_argument('-d', '--lam_data', type=float, default=1.0, required=False)
    parser.add_argument('-a', '--lam_align', type=float, default=0.02, required=False)
    parser.add_argument('-j', '--lam_joint_cross', type=float, default=1.0, required=False)
    parser.add_argument('-r', '--lam_real_cross', type=float, default=1.0, required=False)
    parser.add_argument('-s', '--lam_sup', type=float, default=0.0, required=False)
    parser.add_argument('-x', '--lam_klcross', type=float, default=0.0, required=False)
    parser.add_argument('-p', '--lam_poe', type=float, default=1.0, required=False)
    parser.add_argument('-t', '--lam_contrastive', type=float, default=0.0, required=False)
    parser.add_argument('--lam_contrast_samp', type=float, default=0.0, required=False)
    parser.add_argument('--perturb_prop', type=float, default=0.25, required=False)
    parser.add_argument('-o', '--lam_orthog', type=float, default=1.0, required=False)
    parser.add_argument('-l', '--lr', type=float, default=2e-3, required=False)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--dgex', type=int, default=49386, required=False)
    parser.add_argument('--dgen', type=int, default=935, required=False)
    parser.add_argument('--dmet', type=int, default=148, required=False)
    parser.add_argument('--dprot', type=int, default=6496, required=False)

    args = parser.parse_args()

    fold = args.fold
    dgex = args.dgex
    dgen = args.dgen
    dmet = args.dmet
    dprot = args.dprot

    datadir = args.datadir

    ######################
    ## Loading the data ##
    ######################
    gex = ad.read_h5ad(os.path.join(datadir,'gex-pp.h5ad'))
    gen = ad.read_h5ad(os.path.join(datadir,'gen-pp.h5ad'))
    met = ad.read_h5ad(os.path.join(datadir,'met-pp.h5ad'))
    prot = ad.read_h5ad(os.path.join(datadir,'prot-pp.h5ad'))

    # Selecting the correct feature space to start with
    all_data_sub = [gex, gen, met, prot]
    chosen_dims = [dgex, dgen, dmet, dprot]
    chosen_feats = []
    for i, d in enumerate(all_data_sub):
        if chosen_dims[i] == d.shape[1]:
            print('using original features for modality ' + str(i))
            chosen_feats.append(d.X)
        else:
            chosen_feats.append(d.obsm['X_pca'][:,:chosen_dims[i]])

    gex = ad.AnnData(X=chosen_feats[0],\
                         obs=gex.obs,\
                         obsm={"X_pca":chosen_feats[0]},\
                         layers={"counts":chosen_feats[0]})
    gen = ad.AnnData(X=chosen_feats[1],\
                             obs=gen.obs,\
                             obsm={"X_pca":chosen_feats[1]},\
                             layers={"counts":chosen_feats[1]})
    met = ad.AnnData(X=chosen_feats[2],\
                          obs=met.obs,\
                          obsm={"X_pca":chosen_feats[2]},\
                          layers={"counts":chosen_feats[2]})
    prot = ad.AnnData(X=chosen_feats[3],\
                      obs=prot.obs,\
                      obsm={"X_pca":chosen_feats[3]},\
                      layers={"counts":chosen_feats[3]})

    #######################
    ## Configure dataset ##
    #######################
    splitname = 'split'+str(fold)
    gex_train = gex[gex.obs[splitname] == 'train']
    gen_train = gen[gen.obs[splitname] == 'train']
    met_train = met[met.obs[splitname] == 'train']
    prot_train = prot[prot.obs[splitname] == 'train']

    jasmine.models.configure_dataset(
        gex_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        gen_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        met_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        prot_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )

    gex_test = gex[gex.obs[splitname] == 'test']
    gen_test = gen[gen.obs[splitname] == 'test']
    met_test = met[met.obs[splitname] == 'test']
    prot_test = prot[prot.obs[splitname] == 'test']

    jasmine.models.configure_dataset(
        gex_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        gen_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        met_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        prot_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )

    ###########
    ## Train ##
    ###########

    # Model Params
    latent_dim = 64 # 4  # 50
    x2u_h_depth = 2
    x2u_h_dim = 512
    u2x_h_depth = 1  # 1
    u2x_h_dim = 256  # 256
    du_h_depth = 1
    du_h_dim = 256  # 256
    dropout = 0.2

    lam_data = args.lam_data #1.0
    lam_kl = args.lam_kl # 0.97  # 0.3
    lam_align = args.lam_align # 0.868  # 0.02
    lam_joint_cross = args.lam_joint_cross # 1.0  # 1.0
    lam_real_cross = args.lam_real_cross # 1.0  # 1.0
    lam_poe = args.lam_poe # 1.0
    lam_klcross = args.lam_klcross # 0.0
    lam_cos = args.lam_cos # 0.2858  # 0.02
    lam_sup = args.lam_sup # 0.0
    lam_contrastive = args.lam_contrastive # 0.0
    lam_contrast_samp = args.lam_contrast_samp # 0.0
    perturb_prop = args.perturb_prop # 0.0
    lam_orthog = args.lam_orthog
    lr = args.lr # 0.0

    temperature = 0.1
    normalize_u = True
    random_seed = 2

    mlp_h_depth = 1
    mlp_h_dim = 256
    embedding_dim = 100
    projection_dim = 100
    n_head = 4
    n_layer = 2
    batch_first = True
    norm_first = True


    # Build and compile model
    print('Building model...')
    model = jasmine.models.JASMINEModel(
        {"gex": gex_train, "gen": gen_train, 'met': met_train, 'prot': prot_train},
        latent_dim=latent_dim,
        x2u_h_depth=x2u_h_depth,
        x2u_h_dim=x2u_h_dim,
        u2x_h_depth=u2x_h_depth,
        u2x_h_dim=u2x_h_dim,
        du_h_depth=du_h_depth,
        du_h_dim=du_h_dim,
        dropout=dropout,
        shared_batches=False,
        random_seed=random_seed
    )
    
    print('Compiling model...')
    model.compile(
        lam_data=lam_data, lam_kl=lam_kl, lam_align=lam_align,
        lam_joint_cross=lam_joint_cross, lam_real_cross=lam_real_cross, lam_cos=lam_cos, normalize_u=normalize_u,
        lam_poe=lam_poe, lam_klcross=lam_klcross, lam_sup=lam_sup,
        lam_contrastive=lam_contrastive, temperature=temperature,
        lam_contrast_samp=lam_contrast_samp, perturb_prop=perturb_prop,
        lam_orthog=lam_orthog, lr=lr,
        modality_weight={"gex": 1, "gen": 1, 'met': 1, 'prot': 1}, latent_dim=latent_dim
    )   #change domain_weight to modality_weight


    # Train and save model
    print('Training model...')
    model.fit(
        {"gex": gex_train, "gen": gen_train, 'met': met_train, 'prot': prot_train}
    )

    ################
    ## Save Model ##
    ################
    modeldir = args.modeldir
    modelname = 'pretrain_JASMINE_adni_k'+str(lam_kl)+\
                                     '_c'+str(lam_cos)+\
                                     '_d'+str(lam_data)+\
                                     '_a'+str(lam_align)+\
                                     '_j'+str(lam_joint_cross)+\
                                     '_r'+str(lam_real_cross)+\
                                     '_s'+str(lam_sup)+\
                                     '_x'+str(lam_klcross)+\
                                     '_p'+str(lam_poe)+\
                                     '_t'+str(lam_contrastive)+\
                                     '_ts'+str(lam_contrast_samp)+\
                                     '_pp'+str(perturb_prop)+\
                                     '_o'+str(lam_orthog)+\
                                     '_l'+str(lr)+\
                                     '_dgex'+str(dgex)+\
                                     '_dgen'+str(dgen)+\
                                     '_dmet'+str(dmet)+\
                                     '_dprot'+str(dprot)+\
                                     '_'+str(fold)+\
                                     '.dill'
    model.save(os.path.join(modeldir,modelname))

if __name__ == "__main__":
    main()
