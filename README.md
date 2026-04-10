# JASMINE

JASMINE (**J**oint **A**nd modality-**S**pecific **M**ultimodal representation learning handling **IN**compl**E**te data) is a self-supervised representation learning method that generates compact, task-agnostic embeddings that integrate multi-omics data while handling missing modalities.

![Method schematic](https://github.com/jballard28/JASMINE/blob/main/imgs/jasmine_schematic.png)


## Training JASMINE
JASMINE is trained using scripts with setup similar to those in `train`. A running example on a sample of simulated data provided in `example_data` can be found in `running_scripts/run_example_train_jasmine.sh`:

```
python3 ../train/example_train.py \
     -c 3.0 \
     -j 1.0 \
     -r 1.0 \
     -p 1.0 \
     -t 0.5 \
     --lam_contrast_samp 0.5 \
     --perturb_prop 0.25 \
     -o 0.5 \
     -l 0.002 \
     --nmod 4 \
     --nsamp 100 \
     --missing_setting mask_mar_i \
     --fold 1 \
     --datadir ../example_data \
     --maskdir ../example_data/missing_masks \
     --modeldir ../saved_models \
```
* `-c`: Cosine loss weight
* `-j`: Self loss weight
* `-r`: Cross loss weight
* `-p`: Joint loss weight
* `t`: MCL loss weight
* `--lam_contrast_samp`: SCL loss weight
* `--perturb_prop`: Proportion of features to randomly perturb for SCL
* `-o`: Orthogonality constraints loss weight
* `-l`: Learning rate
* `--nmod`: Number of modalities (for simulated data)
* `--nsamp`: Number of samples (for simulated data)
* `--missing_setting`: Missingness setting for simulated data (MAR, MCAR, MNAR). All provided settings can be found in `example_data/missing_masks`.
* `--fold`: Which train/val/test fold to train on
* `--datadir`: Location of data
* `--maskdir`: Location of missingness masks for simulated data
* `--modeldir`: Where to save trained models

## Embedding generation
Embeddings can be generated from a trained model and saved using scripts similar to those in `embed`. These scripts save the training, validation, and test set embeddings along with their labels to a `.pkl` file. A running example on the sample of simulated data provided in `example_data` can be found in `running_scripts/run_example_embed_jasmine.sh`:

```
python3 ../embed/example_embed.py \
      -c 3.0 \
      -j 1.0 \
      -r 1.0 \
      -p 1.0 \
      -t 0.5 \
      --lam_contrast_samp 0.5 \
      --perturb_prop 0.25 \
      -o 0.5 \
      -l 0.002 \
      --nmod 4 \
      --nsamp 100 \
      --missing_setting mask_mar_i \
      --fold 1 \
      --datadir ../example_data \
      --maskdir ../example_data/missing_masks \
      --modeldir ../saved_models \
      --outdir ../saved_embeddings \
      --test_complete True \
```
Note: The parameter settings above must match those that were used to train the model.
* `--outdir`: Where to save the embeddings
* `--test_complete`: If `True`, test on only the complete samples; else, test on all test samples, including those that are incomplete.

## Evaluation on downstream classification task
Once the embeddings have been generated, they are ready for application to downstream tasks. For example, we can perform classification on the representations generated from the simulated data. An example script for performing this evaluation is provided in `eval/example_eval.py`. This script also saves the summary of results to a `.csv` as well as all raw performance metrics for all data folds to a `.pickle` file. It can be run using the script in `running_scripts/run_example_eval_jasmine.sh`:

```
python3 ../eval/example_eval.py \
     -c 3.0 \
     -j 1.0 \
     -r 1.0 \
     -p 1.0 \
     -t 0.5 \
     --lam_contrast_samp 0.5 \
     --perturb_prop 0.25 \
     -o 0.5 \
     -l 0.002 \
     --nfolds 1 \
     --nmod 4 \
     --nsamp 100 \
     --missing_setting mask_mar_i \
     --embeddir ../saved_embeddings \
     --resultdir ../eval_results \
     --test_complete True \
```
Note: The parameter settings above must match those that were used to train the model. The `test_complete` setting should also match that which was used to generate the embeddings.
* `--embeddir`: Location of the saved embeddings
* `--resultdir`: Where to save the result files
