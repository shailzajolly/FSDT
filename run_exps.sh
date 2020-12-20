#!/bin/bash

#############################

echo "1. Training using whole data"

python run_t5.py --num_train_epochs 10 --is_train True --output_dir t5_ckpts/t5_small/e2e_100par/ --dataset data/e2e_t5data_low.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_100par/

echo "1. Inference on test data"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_100par/ --outfile e2e_100par.txt

#############################

echo "2. Training using 5 percent data"

python run_t5.py --num_train_epochs 15 --is_train True --output_dir t5_ckpts/t5_small/e2e_5par/ --dataset data/e2e_t5data_fs_low.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_5par/

echo "2. Inference on test data"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_5par/ --outfile e2e_5par.txt

#############################

echo "3. Training using 1 percent data"

python run_t5.py --num_train_epochs 20 --is_train True --output_dir t5_ckpts/t5_small/e2e_1par/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par/

echo "3. Inference on test data"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_1par/ --outfile e2e_1par.txt

#############################

echo "4. Generation of 4 percent pseudo parallel"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_1par/ --outfile pseudopar4_30ep_1par.txt --data_variant gen_psd_4par


echo "4. Training using 1 percent parallel data and 4 percent pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train True --output_dir t5_ckpts/t5_small/e2e_1par_4psd/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par.txt
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd/

echo "4. Inference on test data"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_1par_4psd/ --outfile e2e_1par_4psd.txt

#############################

echo "5. Hill Climbing on 4 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small/e2e_1par/ --input_mr_ref data/e2e_t5data_fs_low.json --input_psd_ref t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC.json


echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train True --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd_hc/

echo "5. Inference on test data"

python run_t5.py --is_train False --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc/ --outfile e2e_1par_4psd_hc.txt



echo "All experiments done!"