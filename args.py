import argparse
from utils import get_checkpoint_callback, LoggingCallback


def get_model_args():

    parser = argparse.ArgumentParser(description="model parameters")

    ## Data paths

    parser.add_argument('--output_dir', type=str, default="",
                        help='Output directory path to store checkpoints.')

    parser.add_argument('--outfile', type=str, default="",
                        help='Output data filepath for predictions on test data.')

    parser.add_argument('--dataset', type=str, default="",
                        help='Path of dataset used to train/test t5 model')

    parser.add_argument('--augment_data_filepath', type=str, default="",
                        help='Augment data generated from hill climbing or pseudo parallel samples')

    parser.add_argument('--data_variant', type=str, default="",
                        help='Data augmentation experiments. Possible inputs: 1par, gen_psd_4par, 1par_4psd, 1par_4psd_hc')

    parser.add_argument('--is_train', type=bool, default=True,
                        help='Trains the model if True. Puts model in eval mode if False ')

    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='number of train epochs')

    ## Model building and trainer args

    parser.add_argument('--model_name_or_path', type=str, default="t5-small",
                        help='model_name_or_path.')

    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-small",
                        help='tokenizer_name_or_path.')

    parser.add_argument('--max_input_length', type=int, default=60,
                        help='Input length of model')

    parser.add_argument('--max_output_length', type=int, default=120,
                        help='Output length of model')

    parser.add_argument('--freeze_encoder', type=bool, default=False,
                        help='Freeze encoder params')

    parser.add_argument('--freeze_embeds', type=bool, default=False,
                        help='Freeze embeddings')

    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')

    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='Adam epsilon')

    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps')

    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='Train batch size')

    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Eval batch size')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')

    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Internal hidden size of model.')

    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume checkpoint path')

    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Val check interval')

    parser.add_argument('--n_train', type=int, default=-1,
                        help='Number of training samples to use')

    parser.add_argument('--n_val', type=int, default=1000,
                        help='Number of validation samples to use')

    parser.add_argument('--n_test', type=int, default=-1,
                        help='Number of test samples to use')

    parser.add_argument('--early_stop_callback', type=bool, default=False)

    parser.add_argument('--fp_16', type=bool, default=False,
                        help="if you want to enable 16-bit training then install apex and set this to true")

    parser.add_argument('--opt_level', type=str, default='O1',
                        help='you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties')

    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='if you enable 16-bit training then set this to a sensible value, 0.5 is a good default')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generator')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))
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

def get_hc_args():
    parser = argparse.ArgumentParser(description="hill climbing args")

    parser.add_argument('--input_mr_ref', type=str, required=True,
                        help='Input MR file for Hill Climb file.')

    parser.add_argument('--input_psd_ref', type=str, required=True,
                        help='Input PSD_REF file for Hill Climb file.')

    parser.add_argument('--output_file_hc', type=str, default="4perc_mr_pseudoref_HC.json",
                        help='Output file for Hill Climb file.')




