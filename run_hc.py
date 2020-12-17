import json
import math

from scoring_algos import SimulatedAnnealing, HillClimbing
from editor import RobertaEditor
from generator import T5FineTuner
from args import get_model_args, get_train_args, get_sa_args

if __name__=="__main__":

    model_args = get_model_args()
    trainer_args = get_train_args(model_args)
    sa_args = get_sa_args()

    editor  = RobertaEditor()
    generator = T5FineTuner(model_args)
    generator.load_checkpoint(model_args.output_dir)
    print("Model Built")
    generator.cuda()

    input_file = 't5_ckpts/pseudo_par_files/4perc_mr_pseudoref.json'
    output_file_hc = '4perc_mr_pseudoref_HC1.json'
    
    hill_climb = HillClimbing(generator)
    hill_climb.adding_missing_slotvalues(input_file, output_file_hc)

