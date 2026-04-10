import os
import sys

import anndata as ad
import scanpy as sc
import scglue
import numpy as np
import pandas as pd
import pickle
import math

sys.path.insert(0,'../model')
import jasmine
import jasmine.models

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, balanced_accuracy_score, f1_score, \
    roc_auc_score, average_precision_score

import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddir', type=str, required=True)
    parser.add_argument('--resultdir', type=str, required=True)
    parser.add_argument('--missing_setting', type=str, default=None, required=False)
    parser.add_argument('--test_complete', type=str, required=True)
    parser.add_argument('--val', type=str, default='False', required=False)
    parser.add_argument('--nfolds', type=int, default=5, required=False)
    parser.add_argument('--nmod', type=int, default=4) # 2,3,4,5,6
    parser.add_argument('--nsamp', type=int, default=3000) # 3000, 1500
    parser.add_argument('-k', '--lam_kl', type=float, default=0.97, required=False)
    parser.add_argument('-c', '--lam_cos', type=float, default=3.0, required=False)
    parser.add_argument('-d', '--lam_data', type=float, default=1.0, required=False)
    parser.add_argument('-a', '--lam_align', type=float, default=0.0, required=False)
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
    
    args = parser.parse_args()
    
    if args.test_complete == 'True':
        test_complete = True
    elif args.test_complete == 'False':
        test_complete = False
    else:
        sys.exit('Invalid value for test_complete. Must be True or False.')
    print('test_complete: ' + str(test_complete))
    
    if args.val == 'True':
        val = True
    elif args.val == 'False':
        val = False
    else:
        sys.exit('Invalid value for val. Must be True or False.')
   
    nfolds = args.nfolds 
    folds = list(range(1,nfolds+1))
    nmod = args.nmod
    nsamp = args.nsamp
    missing_setting = args.missing_setting
    print('missing setting: ' + str(missing_setting))
    
    var_dict = {}
    
    #############
    ## JASMINE ##
    #############
    # Classification metrics
    acc = []
    bal_acc = []
    precision = []
    recall = []
    f1 = []
    auroc = []
    ap = []
    
    for fold in folds:
    
        print('fold ' + str(fold))

        # Model params
        lam_poe = args.lam_poe # 1.0
        lam_kl = args.lam_kl # 0.97  # 0.3
        lam_cos = args.lam_cos # 3.0  # 0.02
        lam_klcross = args.lam_klcross # 0.0
        lam_data = args.lam_data # 1.0
        lam_align = args.lam_align # 0.0  # 0.02
        lam_joint_cross = args.lam_joint_cross # 1.0  # 1.0
        lam_real_cross = args.lam_real_cross # 1.0  # 1.0
        lam_sup = args.lam_sup # 0.0
        lam_contrastive = args.lam_contrastive # 1.0 # 2.0
        lam_contrast_samp = args.lam_contrast_samp # 0.5
        perturb_prop = args.perturb_prop # 0.25
        lam_orthog = args.lam_orthog # 1.0
        lr = args.lr # 0.002
    
        # Load the embeddings
        embeddir = args.embeddir
        if test_complete:
            embedfname = 'JASMINE_example_embed_complete_k'+str(lam_kl)+\
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
            embedfname = 'JASMINE_example_embed_incomplete_k'+str(lam_kl)+\
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
        with open(os.path.join(embeddir, embedfname), 'rb') as f:
            embed = pickle.load(f)


        # Train data
        X_embed_train = embed['train_embed']
        y_train = embed['label_train']
    
        # Test only
        if val:
            X_embed_test = embed['val_embed']
            y_test = embed['label_val']
        else:
            X_embed_test = embed['test_embed']
            y_test = embed['label_test']
            
    
        # -----------------------------------------------------------------
        
        # Classification - XGBoost
    
        le = LabelEncoder()
        model_joint = XGBClassifier(n_estimators=300,
                                 learning_rate=0.5,
                                 max_depth=10,
                                 objective='multi:softprob',
                                 tree_method="hist",
                                 device="cuda")
        #                          eval_metric=log_loss)
        model_joint.fit(X_embed_train, le.fit_transform(y_train))
        ypred_joint = model_joint.predict(X_embed_test)
        ypred_joint = le.inverse_transform(ypred_joint)
        pred_probs = model_joint.predict_proba(X_embed_test)
        
        report = classification_report(y_test, ypred_joint, zero_division=0.0, output_dict=True)
    
        # indices of classes that are present in test set
        intest = list(set(np.searchsorted(model_joint.classes_,y_test)))
        # subset and renormalize
        pred_probs_sub = pred_probs[:,intest]/pred_probs[:,intest].sum(axis=1,keepdims=1)
    
        # AUC
        macro_roc_auc_ovr = roc_auc_score(
            y_test,
            pred_probs_sub,
            multi_class="ovr",
            average="macro",
        )
    
        # Average precision score ('This implementation is not interpolated and is different from computing the area under the precision-recall curve with the trapezoidal rule, which uses linear interpolation and can be too optimistic.')
        avg_precision_score = average_precision_score(y_test, pred_probs_sub,
                                                     average='macro')
    
    
        acc.append(report['accuracy'])
        bal_acc.append(balanced_accuracy_score(y_test, ypred_joint))
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        f1.append(report['macro avg']['f1-score'])
        auroc.append(macro_roc_auc_ovr)
        ap.append(avg_precision_score)
    
    print('acc: ' + str(acc))
    print('bal_acc: ' + str(bal_acc))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f1: ' + str(f1))
    print('auroc: ' + str(auroc))
    print('ap: ' + str(ap))
    
    jasmine_results = {'classification':[acc, bal_acc, precision, recall, f1, auroc, ap]}
    var_dict['JASMINE'] = jasmine_results
    
    
    
    # Save the results
    clf_metrics = ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'AUROC', 'Avg Precis']
    clf_metrics = ['Balanced Acc', 'AUROC', 'Avg Precis']
    clf_idcs = [1,5,6]
    clf_data = []
    clf_sd = []
    for c in range(len(clf_metrics)):
        temp = [clf_metrics[c]]
        temp_sd = [clf_metrics[c]]
        for m in var_dict.keys():
            temp.append(np.mean(var_dict[m]['classification'][clf_idcs[c]]))
            temp_sd.append(np.std(var_dict[m]['classification'][clf_idcs[c]]))
        clf_data.append(temp)
        clf_sd.append(temp_sd)
    
    df = pd.DataFrame(clf_data, 
                      columns=['Metric']+list(var_dict.keys()))
    df = df.melt(id_vars='Metric',
                   var_name='Method',
                   value_name='Mean')
    
    df_sd = pd.DataFrame(clf_sd,
                      columns=['Metric']+list(var_dict.keys()))
    df_sd = df_sd.melt(id_vars='Metric',
                   var_name='Method',
                   value_name='SD')
    
    df['SD'] = df_sd['SD']
    
    
    resultdir = args.resultdir
    resultdf = pd.concat([group for (name,group) in df.groupby('Metric', sort=False)])
    param_suffix = 'k'+str(lam_kl)+\
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
                    '_'+missing_setting
    
    if test_complete:
        if val:
            result_fname = 'results_val_'+\
                           param_suffix+\
                           '_testcomplete.csv'
        else:
            result_fname = 'results_test_'+\
                           param_suffix+\
                           '_testcomplete.csv'
    else:
        if val:
            result_fname = 'results_val_'+\
                           param_suffix+\
                           '_testincomplete.csv'
        else:
            result_fname = 'results_test_'+\
                           param_suffix+\
                           '_testincomplete.csv'
    
    resultdf.to_csv(os.path.join(resultdir,result_fname))
    
    
    # Save var_dict to pickle
    with open(os.path.join(resultdir,result_fname[:-4]+'.pickle'), 'wb') as handle:
        pickle.dump(var_dict, handle)


if __name__ == "__main__":
    main()
