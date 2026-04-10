source activate jasmine_env

python3 ../train/example_train.py \
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
     --nmod 4 \
     --nsamp 100 \
     --missing_setting mask_mar_i \
     --fold 1 \
     --datadir ../example_data \
     --maskdir ../example_data/missing_masks \
     --modeldir ../saved_models \

