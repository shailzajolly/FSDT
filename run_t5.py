import pytorch_lightning as pl

from generator import T5FineTuner
from args import get_model_args, get_train_args

import os
from transformers import T5Tokenizer


def train_t5(model_args, trainer_args, model):

    if not os.path.exists(model_args.output_dir):
        os.mkdir(model_args.output_dir)

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model)

def test_t5(model_args, model):

    model.load_checkpoint(model_args.output_dir)

    loader = model.test_dataloader()

    fpred = open("test_output_viggo_5perc_25eps.txt", "w")
    # pseudopar1_30epochs_1par.txt
    # test_output_30epochs_1par+1pseudopar.txt

    for batch in iter(loader):
        outs = model.model.generate(
            batch["source_ids"].cuda(),
            attention_mask=batch["source_mask"].cuda(),
            use_cache=True,
            decoder_attention_mask=batch['target_mask'].cuda(),
            max_length=50,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        for pred in [model.tokenizer.decode(ids) for ids in outs]:
            fpred.write(pred + "\n")

    fpred.close()
    print("Test file written!!")


if __name__=='__main__':

    is_train = True

    model_args = get_model_args()
    trainer_args = get_train_args(model_args)

    model = T5FineTuner(model_args)
    print("Model Built")

    if is_train:
        train_t5(model_args, trainer_args, model)
    else:
        test_t5(model_args, model)
