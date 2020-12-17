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
    editor.cuda()
    generator = T5FineTuner(model_args)
    generator.load_checkpoint(model_args.output_dir)
    print("Model Built")
    generator.cuda()

    simulated_annealing = SimulatedAnnealing(editor, generator, sa_args.t_init, sa_args.C, sa_args.fluency_weight,
                                             sa_args.semantic_weight, sa_args.max_steps)

    data = json.load(open("4perc_mr_pseudoref_HC1.json", "r"))
    batch_size = 32
    num_batches = math.ceil(len(data)/float(batch_size))

    sa_outputs = []

    for i in range(num_batches):

        batch_data = data[batch_size*i:batch_size*(i+1)]
        input_batch = list(zip(*[[i["mr"], i["ref"]] for i in batch_data]))
        sa_outputs_batch = simulated_annealing.run(input_batch)
        print([(i, j) for i, j in zip(input_batch[1], sa_outputs_batch)])
        break
        sa_outputs += sa_outputs_batch
