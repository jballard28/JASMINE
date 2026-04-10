source activate jasmine_env

python3 ../eval/example_eval.py \
     -k 0.97 \
     -c 3.0 \
     -d 1.0 \
     -a 0.0 \
     -j 1.0 \
     -r 1.0 \
     -s 0.0 \
     -x 0.0 \
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

