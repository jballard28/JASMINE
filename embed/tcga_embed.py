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

import pickle

def main():

    parser = argparse.ArgumentParser(description='Get JASMINE embeddings for TCGA')
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
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
    jasmine.models.configure_dataset(
        mRNAseq, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid",
        use_cell_type="uid"
    )
    jasmine.models.configure_dataset(
        Methylation, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid",
        use_cell_type="uid"
    )
    jasmine.models.configure_dataset(
        miRNAseq, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid",
        use_cell_type="uid"
    )
    jasmine.models.configure_dataset(
        RPPA, "Normal", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca",
        use_batch="batch", use_obs_names="uid",
        use_cell_type="uid"
    )


    mRNAseq_train = mRNAseq[~mRNAseq.obs.group.isin([fold,6])]
    Methylation_train = Methylation[~Methylation.obs.group.isin([fold,6])]
    miRNAseq_train = miRNAseq[~miRNAseq.obs.group.isin([fold,6])]
    RPPA_train = RPPA[~RPPA.obs.group.isin([fold,6])]

    mRNAseq_test = mRNAseq[mRNAseq.obs.group == fold]
    Methylation_test = Methylation[Methylation.obs.group == fold]
    miRNAseq_test = miRNAseq[miRNAseq.obs.group == fold]
    RPPA_test = RPPA[RPPA.obs.group == fold]


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


    # Build model
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
    
    # Load saved model
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

    print('Adopting pretrained weights...')
    model.adopt_pretrained_model(jasmine.models.load_model(os.path.join(modeldir, modelname)))


    # Get embeddings
    joint_X_train, uid_train = model.encode_sample({"mRNAseq": mRNAseq_train, "Methylation": Methylation_train, 'miRNAseq': miRNAseq_train, 'RPPA': RPPA_train}, lam_poe)
    joint_X_test, uid_test = model.encode_sample({"mRNAseq": mRNAseq_test, "Methylation": Methylation_test, 'miRNAseq': miRNAseq_test, 'RPPA': RPPA_test}, lam_poe)

    self_X_train, _ = model.cross_encode_sample({"mRNAseq": mRNAseq_train, "Methylation": Methylation_train, 'miRNAseq': miRNAseq_train, 'RPPA': RPPA_train}, False)
    self_X_test, _ = model.cross_encode_sample({"mRNAseq": mRNAseq_test, "Methylation": Methylation_test, 'miRNAseq': miRNAseq_test, 'RPPA': RPPA_test}, False)

    # Concatenate joint and self-embeddings
    jointself_X_train = np.concatenate((joint_X_train,self_X_train),axis=1)
    jointself_X_test = np.concatenate((joint_X_test,self_X_test),axis=1)

    train_embed = jointself_X_train
    test_embed = jointself_X_test



    # Save embeddings to output file
    outdir = args.outdir
    outfname = 'JASMINE_tcga_embed_k'+str(lam_kl)+\
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
                                 '.pkl'

    outdict = {'train_embed': train_embed,
               'test_embed': test_embed,
               'uid_train': uid_train,
               'uid_test': uid_test}

    with open(os.path.join(outdir, outfname), 'wb') as f:
        pickle.dump(outdict, f)


if __name__ == "__main__":
    main()
