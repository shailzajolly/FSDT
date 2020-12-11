import argparse
from utils import get_checkpoint_callback, LoggingCallback

args_dict = dict(
    output_dir="", # path to save the checkpoints
    dataset="",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_input_length=60,
    max_output_length=120,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=1,
    n_gpu=1,
    resume_from_checkpoint=None,
    val_check_interval = 1.0,
    n_val=500,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)



def get_model_args():
    args_dict.update({'output_dir': 'e2e_full/', 'num_train_epochs': 10,
                      'dataset': "data/e2e_t5data_low.json",
                      'train_batch_size': 64, 'eval_batch_size': 64})

    args = argparse.Namespace(**args_dict)
    return args

def get_train_args(args):

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=get_checkpoint_callback(args),
        val_check_interval=args.val_check_interval,
        # logger=wandb_logger,
        callbacks=[LoggingCallback()],
    )

    return train_params

def get_sa_args():

    args_dict = {
        "t_init": 3e-2,
        "C": 3e-4,
        "fluency_weight": 3,
        "semantic_weight": 8,
        "max_steps": 20,
    }

    args = argparse.Namespace(**args_dict)
    return args




