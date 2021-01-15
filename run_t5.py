import pytorch_lightning as pl

from generator import T5FineTuner
from args import get_model_args, get_train_args

import os
import time

def train_t5(model_args, trainer_args, model):

    if not os.path.exists(model_args.output_dir):
        os.mkdir(model_args.output_dir)

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model)

def test_t5(model_args, model):


    model.load_checkpoint(model_args.output_dir)
    model.to('cuda')

    loader = model.test_dataloader()
    start_time = time.time()

    fpred = open(model_args.output_dir + model_args.outfile, "w")

    for batch in iter(loader):
        outs = model.model.generate(
            batch["source_ids"].cuda(),
            attention_mask=batch["source_mask"].cuda(),
            use_cache=True,
            decoder_attention_mask=batch['target_mask'].cuda(),
            max_length=model_args.max_output_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        for pred in [model.tokenizer.decode(ids) for ids in outs]:
            fpred.write(pred + "\n")

    fpred.close()
    print("Test file written!")
    
    end_time = time.time()
    print(f"Time elapsed in seconds {end_time-start_time}")

if __name__=='__main__':

    model_args = get_model_args()
    trainer_args = get_train_args(model_args)

    model = T5FineTuner(model_args)
    print("Model Built")

    if model_args.is_train:
        train_t5(model_args, trainer_args, model)
    else:
        test_t5(model_args, model)
