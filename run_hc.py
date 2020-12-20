import json
import math

from scoring_algos import HillClimbing
from generator import T5FineTuner
from args import get_model_args, get_train_args, get_hc_args


if __name__=="__main__":

    model_args = get_model_args()
    trainer_args = get_train_args(model_args)
    hc_args = get_hc_args()

    generator = T5FineTuner(model_args)
    generator.load_checkpoint(model_args.output_dir)
    print("Model Built")
    generator.cuda()
    
    hill_climb = HillClimbing(generator)
    hill_climb.adding_missing_slotvalues(hc_args.input_mr_ref, hc_args.input_psd_ref, hc_args.output_file_hc)

