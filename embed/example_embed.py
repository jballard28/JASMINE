import os
import sys

import torch
import torchvision
import anndata as ad
import networkx as nx
import scanpy as sc
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

    parser = argparse.ArgumentParser(description='train JASMINE model on example simulated data')
    parser.add_argument('--datadir', type=str, default='../example_data', required=False)
    parser.add_argument('--maskdir', type=str, default='../example_data/missing_masks', required=False)
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
    parser.add_argument('--nmod', type=int, required=True)
    parser.add_argument('--nsamp', type=int, required=True)
    parser.add_argument('--missing_setting', type=str, default = None, required=False)
    parser.add_argument('--test_complete', type=str, required=True)

    args = parser.parse_args()

    fold = args.fold
    nmod = args.nmod
    nsamp = args.nsamp
    missing_setting = args.missing_setting
    if args.test_complete == 'True':
        test_complete = True
    elif args.test_complete == 'False':
        test_complete = False
    else:
        sys.exit('Invalid value for test_complete. Must be True or False.')
    

    datadir = args.datadir

    ######################
    ## Loading the data ##
    ######################

    if missing_setting is not None:
        # Loading the missingness masks
        maskdir = args.maskdir
        mask = pd.read_csv(os.path.join(maskdir,f'{missing_setting}.csv'))

    # Loading the data
    allmod = []
    for m in range(nmod):
        mod_temp = ad.read_h5ad(os.path.join(datadir,'simdata_'+str(nsamp)+'_mod'+str(m)+'.h5ad'))

        if missing_setting is not None:
            mask = mod_temp.obs[['uid']].merge(mask, how='inner', on='uid')
            mod_mask = mask['mod'+str(m)]
            mod_temp = mod_temp[mod_mask == True]

        # Standardizing based on train means and std
        d_train = mod_temp[~mod_temp.obs.group.isin([fold,6])]
        train_mean = np.mean(d_train.X, axis=0)
        train_std = np.std(d_train.X, axis=0)
        mod_temp.X = (mod_temp.X-train_mean)/(train_std)

        # Create new fields in anndata objects
        mod_temp.obsm['X_pca'] = mod_temp.X
        mod_temp.layers['counts'] = mod_temp.X.copy()

        allmod.append(mod_temp)
        print('Loaded mod'+str(m)+'.')

    # Find the uids that have all modalities
    mod_uids = [m.obs['uid'] for m in allmod]
    
    uid_allmod = set(mod_uids[0])
    alluids = set(mod_uids[0])
    for i in range(nmod-1):
        uid_allmod = uid_allmod & set(mod_uids[i+1])
        alluids = alluids | set(mod_uids[i+1])
    uid_allmod = list(uid_allmod)
    alluids = list(alluids)
    
    # Data with all samples
    mod_allsamp = [m.copy() for m in allmod]
    
    # Data with samples that only have all modalities
    mod_sub = [m[m.obs.uid.isin(uid_allmod)].copy() for m in allmod]
    
    #######################
    ## Configure dataset ##
    #######################

    # Training data
    data_train = [m[~m.obs.group.isin([fold,6])] for m in mod_allsamp]
    # Configure datasets
    for m in data_train:
        jasmine.models.configure_dataset(
            m, "Normal", use_highly_variable=False,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_obs_names="uid",
            use_cell_type='label'
        )

    # Val data
    if test_complete:
        data_val = [m[m.obs.group == fold] for m in mod_sub]
    else:
        data_val = [m[m.obs.group == fold] for m in mod_allsamp]
    # Configure datasets
    for m in data_val:
        jasmine.models.configure_dataset(
            m, "Normal", use_highly_variable=False,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_obs_names="uid",
            use_cell_type='label'
        )

    # Test data
    if test_complete:
        data_test = [m[m.obs.group == 6] for m in mod_sub]
    else:
        data_test = [m[m.obs.group == 6] for m in mod_allsamp]
    # Configure datasets
    for m in data_test:
        jasmine.models.configure_dataset(
            m, "Normal", use_highly_variable=False,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_obs_names="uid",
            use_cell_type='label'
        )

    # list of modality names
    mod_names = ['mod'+str(i) for i in range(nmod)]


    # Model Params
    model_name = 'JASMINE'
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
        {mod_names[i]:data_train[i] for i in range(nmod)},
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
    modelname = 'pretrain_JASMINE_example_k'+str(lam_kl)+\
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
                                        '_nmod'+str(nmod)+\
                                        '_nsamp'+str(nsamp)+\
                                        '_'+missing_setting+\
                                        '_'+str(fold)+\
                                        '.dill'

    print('Adopting pretrained weights...')
    model.adopt_pretrained_model(jasmine.models.load_model(os.path.join(modeldir, modelname)))

    # Get embeddings
    joint_X_train, y_train = model.encode_sample({mod_names[i]:data_train[i] for i in range(nmod)}, lam_poe)
    joint_X_val, y_val = model.encode_sample({mod_names[i]:data_val[i] for i in range(nmod)}, lam_poe)
    joint_X_test, y_test = model.encode_sample({mod_names[i]:data_test[i] for i in range(nmod)}, lam_poe)
    
    self_X_train, _ = model.cross_encode_sample({mod_names[i]:data_train[i] for i in range(nmod)}, False)
    self_X_val, _ = model.cross_encode_sample({mod_names[i]:data_val[i] for i in range(nmod)}, False)
    self_X_test, _ = model.cross_encode_sample({mod_names[i]:data_test[i] for i in range(nmod)}, False)
    
    # Concatenate joint and self-embeddings
    jointself_X_train = np.concatenate((joint_X_train,self_X_train),axis=1)
    jointself_X_val = np.concatenate((joint_X_val,self_X_val),axis=1)
    jointself_X_test = np.concatenate((joint_X_test,self_X_test),axis=1)
    
    train_embed = jointself_X_train
    val_embed = jointself_X_val
    test_embed = jointself_X_test


    # Save embeddings to output file
    outdir = args.outdir
    if test_complete:
        outfname = 'JASMINE_example_embed_complete_k'+str(lam_kl)+\
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
                                     '_nmod'+str(nmod)+\
                                     '_nsamp'+str(nsamp)+\
                                     '_'+missing_setting+\
                                     '_'+str(fold)+\
                                     '.pkl'
    else:
        outfname = 'JASMINE_example_embed_incomplete_k'+str(lam_kl)+\
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
                                     '_nmod'+str(nmod)+\
                                     '_nsamp'+str(nsamp)+\
                                     '_'+missing_setting+\
                                     '_'+str(fold)+\
                                     '.pkl'


    outdict = {'train_embed': train_embed,
               'val_embed': val_embed,
               'test_embed': test_embed,
               'label_train': y_train,
               'label_val': y_val,
               'label_test': y_test}

    with open(os.path.join(outdir, outfname), 'wb') as f:
        pickle.dump(outdict, f)

if __name__ == "__main__":
    main()
