import torch
import json
import re
import random
import numpy as np

from transformers import T5Tokenizer
from dataset import T5ScorerDataset


SLOTNAMES_GLOB = {'food', 'pricerange', 'eattype', 'near', 'name', 'familyfriendly', 'customer rating', 'area'}
RULES = {'food': "SV food",
         'pricerange_num': "price range SV",
         'pricerange_nonnum': "SV price range",
         'eattype': "SV",
         'near': "near SV",
         'name': "SV",
         'familyfriendly': "SV family friendly",
         'customer rating': "SV customer rating",
         'area': "in SV area"}

class SimulatedAnnealing:

    def __init__(self, editor, generator, t_init, C, fluency_weight, semantic_weight, max_steps):

        self.editor = editor
        self.generator = generator
        self.t_init = t_init
        self.C = C
        self.fluecy_weight = fluency_weight
        self.semantic_weight = semantic_weight
        self.max_steps = max_steps
        self.t5_data_prep = T5ScorerDataset(self.generator.tokenizer, 60, 120)

    def semantic_scorer(self, mr_embeds, ref_embeds):
        return mr_embeds.mm(ref_embeds.T).max(dim=1).values.min()

    def scorer(self, ref_new, ref_org, mr):

        batch = self.t5_data_prep.get_batch(mr, ref_new)
        fluency_scores = self.generator.fluency_score(batch)


    def run(self, input_batch):

        """
        :param input_batch: tuple([mr], [ref])
        :return:
        """
        mr = input_batch[0]
        ref_org = input_batch[1]

        ref_old = input_batch[1]
        batch_size =  len(mr)

        for t in range(self.max_setps):

            T = max(self.t_init - self.C * t, 0)

            ops = np.random.randint(0, 3, batch_size)
            positions = [random.randint(0,len(i.strip().split(" "))-1) for i in ref_old]

            ref_hat = self.editor.edit(ref_old, ops, positions)



class HillClimbing:

    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer

    def hasNumbers(self, inputString):
        return bool(re.search(r'\d', inputString))

    def insert_missingslot(self, mr, ref, missing_slot, t5_data_prep):
        hypothesis = []
        tokens = ref.split()
        for i in range(len(tokens)):
            hypothesis.append(" ".join(tokens[:i] + [missing_slot] + tokens[i:]))

        batch = t5_data_prep.get_batch(mr, hypothesis)
        outs = self.model.fluency_score(batch)
        return hypothesis[torch.max(outs, 0)[1].cpu().item()]

    def adding_missing_slotvalues(self, infile, outfile):

        mr_pseudoref = json.load(open(infile, 'r'))

        t5_data_prep = T5ScorerDataset(self.tokenizer, 60, 120)
        hill_climb_res = []

        for mr_ref in mr_pseudoref:
            temp_dict = {}

            mr = mr_ref[0].strip().lower()
            ref = mr_ref[1].strip().lower()

            slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
            slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out values between []

            slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]
            slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

            missing_slots = []
            coverage = 0

            for sn, sv in zip(slotnames, slotvalues):

                if sv in ['yes', 'no', 'not']:
                    org_sv = sv
                    sv = 'friendly'

                if sv in ref:
                    coverage += 1
                else:
                    if sn == "pricerange" and ("price range" in ref):
                        continue
                    elif sn == "pricerange":
                        if self.hasNumbers(sv):
                            sv = RULES["pricerange_num"].replace("SV", sv)
                        else:
                            sv = RULES["pricerange_nonnum"].replace("SV", sv)
                    elif sn == "familyfriendly":
                        if org_sv in ['no' or 'not']:
                            sv = RULES["familyfriendly"].replace("SV", "not")
                        else:
                            sv = RULES["familyfriendly"].replace("SV ", "")
                    else:
                        sv = RULES[sn].replace("SV", sv)

                    missing_slots.append(sv)
            temp_dict["mr"] = mr

            best_ref = ref
            for missing_slot in missing_slots:
                best_ref = self.insert_missingslot(mr, best_ref, missing_slot, t5_data_prep)
                print(missing_slot)
                print(best_ref)
                print("----")

            temp_dict["ref"] = best_ref
            hill_climb_res.append(temp_dict)
        json.dump(hill_climb_res, open(outfile, 'w+'))

        print("Hill climb results dumped!!")