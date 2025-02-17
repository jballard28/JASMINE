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

    parser = argparse.ArgumentParser(description='train JASMINE model on TCGA')
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
    parser.add_argument('--dmrna', type=int, default=20390, required=False)
    parser.add_argument('--dmeth', type=int, default=13130, required=False)
    parser.add_argument('--dmirna', type=int, default=119, required=False)
    parser.add_argument('--drppa', type=int, default=122, required=False)

    args = parser.parse_args()

    fold = args.fold
    dmrna = args.dmrna
    dmeth = args.dmeth
    dmirna = args.dmirna
    drppa = args.drppa

    datadir = args.datadir 


    ######################
    ## Loading the data ##
    ######################
    mRNAseq = ad.read_h5ad(os.path.join(datadir,'mRNAseq-pp_kfold_multidim.h5ad'))
    Methylation = ad.read_h5ad(os.path.join(datadir,'Methylation-pp_kfold_multidim.h5ad'))
    miRNAseq = ad.read_h5ad(os.path.join(datadir,'miRNAseq-pp_kfold_multidim.h5ad'))
    RPPA = ad.read_h5ad(os.path.join(datadir,'RPPA-pp_kfold_multidim.h5ad'))

    # Selecting the correct feature space to start with
    all_data_sub = [mRNAseq, Methylation, miRNAseq, RPPA]
    chosen_dims = [dmrna, dmeth, dmirna, drppa]
    chosen_feats = []
    for i, d in enumerate(all_data_sub):
        if chosen_dims[i] == d.shape[1]:
            print('using original features for modality ' + str(i))
            chosen_feats.append(d.X)
        else:
            chosen_feats.append(d.obsm['X_pca'][:,:chosen_dims[i]])

    mRNAseq = ad.AnnData(X=chosen_feats[0],\
                         obs=mRNAseq.obs,\
                         obsm={"X_pca":chosen_feats[0]},\
                         layers={"counts":chosen_feats[0]})
    Methylation = ad.AnnData(X=chosen_feats[1],\
                             obs=Methylation.obs,\
                             obsm={"X_pca":chosen_feats[1]},\
                             layers={"counts":chosen_feats[1]})
    miRNAseq = ad.AnnData(X=chosen_feats[2],\
                          obs=miRNAseq.obs,\
                          obsm={"X_pca":chosen_feats[2]},\
                          layers={"counts":chosen_feats[2]})
    RPPA = ad.AnnData(X=chosen_feats[3],\
                      obs=RPPA.obs,\
                      obsm={"X_pca":chosen_feats[3]},\
                      layers={"counts":chosen_feats[3]})

    #######################
    ## Configure dataset ##
    #######################
    mRNAseq_train = mRNAseq[~mRNAseq.obs.group.isin([fold,6])]
    Methylation_train = Methylation[~Methylation.obs.group.isin([fold,6])]
    miRNAseq_train = miRNAseq[~miRNAseq.obs.group.isin([fold,6])]
    RPPA_train = RPPA[~RPPA.obs.group.isin([fold,6])]

    jasmine.models.configure_dataset(
        mRNAseq_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        Methylation_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        miRNAseq_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        RPPA_train, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )

    mRNAseq_test = mRNAseq[mRNAseq.obs.group == fold]
    Methylation_test = Methylation[Methylation.obs.group == fold]
    miRNAseq_test = miRNAseq[miRNAseq.obs.group == fold]
    RPPA_test = RPPA[RPPA.obs.group == fold]

    jasmine.models.configure_dataset(
        mRNAseq_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        Methylation_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        miRNAseq_test, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid"
        # use_cell_type="diagnosis"
    )
    jasmine.models.configure_dataset(
        RPPA_test, "Normal", use_highly_variable=False,
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
        {"mRNAseq": mRNAseq_train, "Methylation": Methylation_train, 'miRNAseq': miRNAseq_train, 'RPPA': RPPA_train},
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
        modality_weight={"mRNAseq": 1, "Methylation": 1, 'miRNAseq': 1, 'RPPA': 1}, latent_dim=latent_dim
    )   #change domain_weight to modality_weight


    # Train and save model
    print('Training model...')
    model.fit(
        {"mRNAseq": mRNAseq_train, "Methylation": Methylation_train, 'miRNAseq': miRNAseq_train, 'RPPA': RPPA_train}
    )

    ################
    ## Save Model ##
    ################
    modeldir = args.modeldir
    modelname = 'pretrain_JASMINE_tcga_k'+str(lam_kl)+\
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
                                     '_dmrna'+str(dmrna)+\
                                     '_dmeth'+str(dmeth)+\
                                     '_dmirna'+str(dmirna)+\
                                     '_drppa'+str(drppa)+\
                                     '_'+str(fold)+\
                                     '.dill'
    model.save(os.path.join(modeldir,modelname))

if __name__ == "__main__":
    main()
