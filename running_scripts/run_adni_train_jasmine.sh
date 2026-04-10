source activate jasmine_env

# New tuning results
python3 ../train/adni_train.py \
     -c 3.0 \
     -j 1.0 \
     -r 1.0 \
     -p 1.0 \
     -t 2.0 \
     --lam_contrast_samp 0.5 \
     --perturb_prop 0.25 \
     -o 0.5 \
     -l 0.002 \
     --fold 1 \
     --datadir /path/to/data/ \
     --modeldir /path/to/saved/models \

