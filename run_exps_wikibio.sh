#!/bin/bash

#############################

echo "1. Training using wikibio 100 samples"

python run_t5.py --num_train_epochs 20 --is_train --output_dir wikibio_T5_ckpts/wikibio_100samples_noNone/ --dataset data/wikibio_100train_noNone.json --n_val 256 --max_input_length 300 --max_output_length 180 --train_batch_size 20 --gradient_accumulation_steps 3 --val_batch_size 20 --eval_batch_size 20 --model_name_or_path t5-base

mv wikibio_T5_ckpts/*.ckpt wikibio_T5_ckpts/wikibio_100samples_noNone/

echo "1. Inference on test data"

python run_t5.py --output_dir wikibio_T5_ckpts/wikibio_100samples_noNone/ --outfile wikibio_100samples_noNone.txt --dataset data/wikibio_100train_noNone.json --model_name_or_path t5-base --max_input_length 300 --max_output_length 180

#############################

echo "2. Training using wikibio 500 samples"

python run_t5.py --num_train_epochs 15 --is_train --output_dir wikibio_T5_ckpts/wikibio_500samples_noNone/ --dataset data/wikibio_500train_noNone.json --n_val 1000 --max_input_length 300 --max_output_length 180 --train_batch_size 20 --gradient_accumulation_steps 3 --val_batch_size 20 --eval_batch_size 20 --model_name_or_path t5-base

mv wikibio_T5_ckpts/*.ckpt wikibio_T5_ckpts/wikibio_500samples_noNone/

echo "2. Inference on test data"

python run_t5.py --output_dir wikibio_T5_ckpts/wikibio_500samples_noNone/ --outfile wikibio_500samples_noNone.txt --dataset data/wikibio_500train_noNone.json --model_name_or_path t5-base --max_input_length 300 --max_output_length 180

#############################

echo "3. Generation of 400 pseudo parallel"

python run_t5.py --dataset data/wikibio_500train_noNone.json --output_dir wikibio_T5_ckpts/wikibio_100samples_noNone/ --outfile pseudopar400_genfrom100ckpt.txt --data_variant gen_psd_4par --data_variant_samples 100 --model_name_or_path t5-base --max_input_length 300 --max_output_length 180

echo "3. Training using 100 parallel and 400 pseudo parallel"

python run_t5.py --num_train_epochs 15 --is_train --output_dir wikibio_T5_ckpts/wikibio_1par_4psd/ --dataset data/wikibio_500train_noNone.json --data_variant 1par_4psd --data_variant_samples 100 --augment_data_filepath wikibio_T5_ckpts/wikibio_100samples_noNone/pseudopar400_genfrom100ckpt.txt --model_name_or_path t5-base --max_input_length 300 --max_output_length 180 --train_batch_size 20 -eval_batch_size 20 --gradient_accumulation_steps 3 --n_val 1000

mv wikibio_T5_ckpts/*.ckpt wikibio_T5_ckpts/wikibio_1par_4psd/

echo "3. Inference on test data"

python run_t5.py --dataset data/wikibio_500train_noNone.json --output_dir wikibio_T5_ckpts/wikibio_1par_4psd/ --outfile wikibio_1par_4psd.txt --model_name_or_path t5-base --max_input_length 300 --max_output_length 180

#############################

echo "4. Hill Climbing on 400 pseudo references"

python run_hc.py --output_dir wikibio_T5_ckpts/wikibio_100samples_noNone/ --input_mr_ref data/wikibio_500train_noNone.json --input_psd_ref wikibio_T5_ckpts/wikibio_100samples_noNone/pseudopar400_genfrom100ckpt.txt --output_file_hc wikibio_T5_ckpts/wikibio_100samples_noNone/pseudopar400_genfrom100ckpt_HC.json --hc_dataset wikibio --hc_num_samples 100 --model_name_or_path t5-base --max_input_length 300 --max_output_length 180

echo "4. Training using 1 percent parallel data and hill climbing outputs"

python run_t5.py --num_train_epochs 15 --is_train --output_dir wikibio_T5_ckpts/wikibio_1par_400psd_hc/ --dataset data/wikibio_500train_noNone.json --data_variant 1par_4psd_hc --data_variant_samples 100 --augment_data_filepath wikibio_T5_ckpts/wikibio_100samples_noNone/pseudopar400_genfrom100ckpt_HC.json --model_name_or_path t5-base --max_input_length 300 --max_output_length 180 --train_batch_size 20 -eval_batch_size 20 --gradient_accumulation_steps 3 --n_val 1000

mv wikibio_T5_ckpts/*.ckpt wikibio_T5_ckpts/wikibio_1par_400psd_hc/

echo "4. Inference on test data"

python run_t5.py --dataset data/wikibio_500train_noNone.json --output_dir wikibio_T5_ckpts/wikibio_1par_400psd_hc/ --outfile wikibio_1par_400psd_hc.txt --model_name_or_path t5-base --max_input_length 300 --max_output_length 180


echo "All experiments done!"

