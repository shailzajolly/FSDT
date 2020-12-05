import pytorch_lightning as pl

from generator import T5FineTuner
from args import get_model_args, get_train_args

import os

model_args = get_model_args()

if not os.path.exists(model_args.output_dir):
    os.mkdir(model_args.output_dir)

trainer_args = get_train_args(model_args)


model = T5FineTuner(model_args)
print("Model Built")

trainer = pl.Trainer(**trainer_args)

trainer.fit(model)
