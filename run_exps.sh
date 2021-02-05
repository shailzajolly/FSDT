#!/bin/bash

#############################

#echo "4. Generation of 4 percent shuffled pseudo parallel"

#python run_t5.py --dataset data/100_shuffled_samples_e2e.json --output_dir t5_ckpts/t5_small/e2e_1par/ --outfile pseudopar4_30ep_1par_shuf.txt --data_variant gen_psd_4par --data_variant_samples 420 --max_input_length 60 --max_output_length 80

#echo "4. Training using 1 percent parallel data and 4 percent shuffled pseudo parallel"

#python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small/e2e_1par_4psd_shuf/ --dataset data/100_shuffled_samples_e2e.json --data_variant 1par_4psd --data_variant_samples 420 --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par_shuf.txt

#mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd_shuf/

#echo "4. Inference on test data"

#python run_t5.py --dataset data/100_shuffled_samples_e2e.json --output_dir t5_ckpts/t5_small/e2e_1par_4psd_shuf/ --outfile e2e_1par_4psd_shuf.txt --max_input_length 60 --max_output_length 80


#############################

#echo "5. Hill Climbing on 4 percent pseudo references shuffled"

#python run_hc.py --output_dir t5_ckpts/t5_small/e2e_1par/ --input_mr_ref data/100_shuffled_samples_e2e.json --input_psd_ref t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par_shuf.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC_shuf.json --hc_dataset e2e --hc_num_samples 420 --model_name_or_path t5-small --max_input_length 60 --max_output_length 80

#echo "5. Training using 1 percent parallel data and hill climbing outputs"

#python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc_shuf/ --dataset data/100_shuffled_samples_e2e.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC_shuf.json

#mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd_hc_shuf/

#echo "5. Inference on test data"

#python run_t5.py --dataset data/100_shuffled_samples_e2e.json --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc_shuf/ --outfile e2e_1par_4psd_hc_shuf.txt --max_input_length 60 --max_output_length 80

#echo "Shuffling exps done!!"
#exit 0

#############################
#echo "1. Running search and learn"

#echo "1. Inference on test data"

#python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc/ --outfile e2e_1par_4psd_hc_4time.txt --model_name_or_path t5-small --max_input_length 60 --max_output_length 50

echo "2. Running search for inference"

echo "2. Hill Climbing on test results"

python run_hc.py --output_dir t5_ckpts/t5_small/e2e_1par/ --input_mr_ref data/e2e_t5data_fs_low.json --input_psd_ref t5_ckpts/t5_small/e2e_1par_4psd/e2e_1par_4psd.txt --output_file_hc t5_ckpts/t5_small/e2e_1par_4psd/e2e_1par_4psd_HCat_inference.txt --hc_dataset e2e --hc_num_samples 420 --model_name_or_path t5-small --max_input_length 60 --max_output_length 100

echo "2. Add two times: time to generate test outputs, time to apply hill climbing on test outputs"

echo "Now Exiting"
exit 0 
#############################

echo "1. Training using whole data"

python run_t5.py --num_train_epochs 10 --is_train --output_dir t5_ckpts/t5_small/e2e_100par/ --dataset data/e2e_t5data_low.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_100par/

echo "1. Inference on test data"

python run_t5.py --output_dir t5_ckpts/t5_small/e2e_100par/ --outfile e2e_100par.txt --dataset data/e2e_t5data_low.json

#############################

echo "2. Training using 5 percent data"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small/e2e_5par/ --dataset data/e2e_t5data_fs_low.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_5par/

echo "2. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_5par/ --outfile e2e_5par.txt

#############################

echo "3. Training using 1 percent data"

python run_t5.py --num_train_epochs 20 --is_train --output_dir t5_ckpts/t5_small/e2e_1par/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par/

echo "3. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_1par/ --outfile e2e_1par.txt

#############################

echo "4. Generation of 4 percent pseudo parallel"


python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_1par/ --outfile pseudopar4_30ep_1par.txt --data_variant gen_psd_4par


echo "4. Training using 1 percent parallel data and 4 percent pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small/e2e_1par_4psd/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par.txt
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd/

echo "4. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_1par_4psd/ --outfile e2e_1par_4psd.txt

#############################

echo "5. Hill Climbing on 4 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small/e2e_1par/ --input_mr_ref data/e2e_t5data_fs_low.json --input_psd_ref t5_ckpts/t5_small/e2e_1par/pseudopar4_30ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC.json


echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/4perc_mr_pseudoref_HC.json
mv t5_ckpts/t5_small/*.ckpt t5_ckpts/t5_small/e2e_1par_4psd_hc/

echo "5. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_fs_low.json --output_dir t5_ckpts/t5_small/e2e_1par_4psd_hc/ --outfile e2e_1par_4psd_hc.txt



echo "All experiments done!"
