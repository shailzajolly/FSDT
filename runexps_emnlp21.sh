#!/bin/bash

#############################

echo "3. Training using 1 percent data"

python run_t5.py --num_train_epochs 20 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --dataset data/e2e_t5data_fs_low.json --data_variant 1par

#mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par/

echo "3. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --outfile e2e_1par.txt

echo "No unparalled done"

#############################
#With 1% unlabeled
#############################


echo "4. Generation of 1 percent pseudo parallel"


python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --outfile pseudopar_1_20ep_1par.txt --data_variant gen_psd_4par --data_variant_samples 420 --data_variant_samples_end 840

echo "4. Training using 1 percent parallel data and 1 percent pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_1psd/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_1_20ep_1par.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_1psd/

echo "4. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_1psd/ --outfile e2e_1par_1psd.txt

#############################

echo "5. Hill Climbing on 1 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --input_mr_ref data/e2e_t5data_low.json --input_psd_ref t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_1_20ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/pseudopar_1_20ep_1parHC.txt

echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_1psd_hc/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar_1_20ep_1parHC.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_1psd_hc/

echo "5. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_1psd_hc/ --outfile e2e_1par_1_hc.txt

echo "1% unparalled done"

#############################
#############################

#############################
#With 3% unlabelled
#############################


echo "4. Generation of 3 percent pseudo parallel"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --outfile pseudopar_3_20ep_1par.txt --data_variant gen_psd_4par --data_variant_samples 420 --data_variant_samples_end 1680

echo "4. Training using 1 percent parallel data and 3 percent pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_3psd/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_3_20ep_1par.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_3psd/

echo "4. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_3psd/ --outfile e2e_1par_3psd.txt

#############################

echo "5. Hill Climbing on 3 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --input_mr_ref data/e2e_t5data_low.json --input_psd_ref t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_3_20ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/pseudopar_3_20ep_1parHC.txt

echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_3psd_hc/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar_3_20ep_1parHC.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_3psd_hc/

echo "5. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_3psd_hc/ --outfile e2e_1par_3_hc.txt

echo "3% unparalled done"

#############################
#############################

#############################
#With 5% unlabelled
#############################


echo "4. Generation of 5 percent pseudo parallel"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --outfile pseudopar_5_20ep_1par.txt --data_variant gen_psd_4par --data_variant_samples 420 --data_variant_samples_end 2520

echo "4. Training using 1 percent parallel data and 5 percent pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_5psd/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_5_20ep_1par.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_5psd/

echo "4. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_5psd/ --outfile e2e_1par_5psd.txt

#############################

echo "5. Hill Climbing on 3 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --input_mr_ref data/e2e_t5data_low.json --input_psd_ref t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_5_20ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/pseudopar_5_20ep_1parHC.txt

echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_5psd_hc/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar_5_20ep_1parHC.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_5psd_hc/

echo "5. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_5psd_hc/ --outfile e2e_1par_5_hc.txt

echo "5% unparalled done"

#############################
#############################

#############################
#With 99% unlabelled
#############################


echo "4. Generation of 99 percent pseudo parallel"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --outfile pseudopar_99_20ep_1par.txt --data_variant gen_psd_4par --data_variant_samples 420 --data_variant_samples_end 0

echo "4. Training using 1 percent parallel data and 99 percent pseudo parallel"

python run_t5.py --num_train_epochs 10 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_99psd/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd  --augment_data_filepath t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_99_20ep_1par.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_99psd/

echo "4. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_99psd/ --outfile e2e_1par_99psd.txt

#############################

echo "5. Hill Climbing on 99 percent pseudo references"

python run_hc.py --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par/ --input_mr_ref data/e2e_t5data_low.json --input_psd_ref t5_ckpts/t5_small_emnlp21/e2e_1par/pseudopar_99_20ep_1par.txt --output_file_hc t5_ckpts/t5_small/e2e_1par/pseudopar_99_20ep_1parHC.txt

echo "5. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 10 --is_train --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_99psd_hc/ --dataset data/e2e_t5data_low.json --data_variant 1par_4psd_hc  --augment_data_filepath t5_ckpts/t5_small/e2e_1par/pseudopar_99_20ep_1parHC.txt

# mv t5_ckpts/t5_small_emnlp21/*.ckpt t5_ckpts/t5_small_emnlp21/e2e_1par_99psd_hc/

echo "5. Inference on test data"

python run_t5.py --dataset data/e2e_t5data_low.json --output_dir t5_ckpts/t5_small_emnlp21/e2e_1par_99psd_hc/ --outfile e2e_1par_99_hc.txt

echo "99% unparalled done"

#############################
#############################

echo "All experiments done!"
