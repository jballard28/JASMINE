source activate jasmine_env

# New tuning results
python3 ../embed/adni_embed.py \
     -k 0.97 \
     -c 3.0 \
     -d 1.0 \
     -a 0.0 \
     -j 1.0 \
     -r 1.0 \
     -s 0.0 \
     -x 0.0 \
     -p 1.0 \
     -t 2.0 \
     --lam_contrast_samp 0.5 \
     --perturb_prop 0.25 \
     -o 0.5 \
     -l 0.002 \
     --fold 1 \
     --datadir /path/to/data/ \
     --modeldir /path/to/saved/models \
     --outdir /path/to/saved/embeddings \

