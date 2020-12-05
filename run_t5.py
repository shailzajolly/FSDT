import pytorch_lightning as pl

from generator import T5FineTuner
from args import get_model_args, get_train_args


model_args = get_model_args()
trainer_args = get_train_args(model_args)


model = T5FineTuner(model_args)

trainer = pl.Trainer(**trainer_args)

trainer.fit(model)