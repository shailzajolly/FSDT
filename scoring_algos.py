import torch
import json
import re
import random
import tqdm
import numpy as np

from transformers import T5Tokenizer
from dataset import T5ScorerDataset


SLOTNAMES_GLOB = {'food', 'pricerange', 'eattype', 'near', 'name', 'familyfriendly', 'customer rating', 'area'}
FF_TYPES = ["kid", "family", "child"]
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
        self.fluency_weight = fluency_weight
        self.semantic_weight = semantic_weight
        self.max_steps = max_steps
        self.t5_data_prep = T5ScorerDataset(self.generator.tokenizer, 60, 120)

    def mr_to_keywords(self, mrs):
        for mr in mrs:
            slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
            slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [

            slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]
            slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

            keywords = []
            for sn, sv in zip(slotnames, slotvalues):

                if sv=="yes":
                    sv = "family-friendly"
                elif sv in ['no', 'not']:
                    sv = "not family-friendly"

                keywords.append(sv)

            yield " ".join(keywords)

    def semantic_scorer(self, mrs, refs):
        mr_embeds = self.editor.get_contextual_word_embeddings(list(self.mr_to_keywords(mrs)))
        ref_embeds = self.editor.get_contextual_word_embeddings(refs)

        return mr_embeds.bmm(ref_embeds.permute(0,2,1)).max(dim=2).values.min(dim=1).values

    def scorer(self, ref_news, mrs):

        batch = self.t5_data_prep.get_batch(mrs, ref_news)
        fluency_scores = self.generator.fluency_score(batch)
        semantic_scores = self.semantic_scorer(mrs, ref_news)
        total_scores = fluency_scores.pow(self.fluency_weight) * semantic_scores.pow(self.semantic_weight)

        return total_scores

    def acceptance_prob(self, ref_hats, ref_olds, mrs, T):
        accept_hat = torch.exp(self.scorer(ref_hats, mrs) - self.scorer(ref_olds, mrs) / T)
        return accept_hat.clamp(0.0, 1.0).squeeze().cpu().detach().numpy().tolist()

    def run(self, input_batch):

        """
        :param input_batch: List([mrs], [refs])
        :return:
        """
        mrs = list(input_batch[0])
        ref_orgs = list(input_batch[1])

        ref_olds = list(input_batch[1])
        batch_size =  len(mrs)
        for t in range(self.max_steps):
            T = max(self.t_init - self.C * t, 0)
            
            ops = np.random.randint(0, 3, batch_size)
            positions = [random.randint(0,len(i.strip().split(" "))-1) for i in ref_olds]

            ref_hats = self.editor.edit(ref_olds, ops, positions)
            accept_probs = self.acceptance_prob(ref_hats.tolist(), ref_olds, mrs, T)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob==1.0:
                    ref_olds[idx] = ref_hats[idx]

        return ref_olds


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

        for mr_ref in tqdm.tqdm(mr_pseudoref):
            temp_dict = {}

            mr = mr_ref[0].strip().lower()
            ref = mr_ref[1].strip().lower()

            slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
            slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [

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
                            sv = RULES["familyfriendly"].strip().replace("SV", "not")
                        else:
                            sv = RULES["familyfriendly"].strip().replace("SV ", "")
                        sv  = "It is " + sv.replace("family", random.sample(FF_TYPES, 1)[0]) +  "."
                    else:
                        sv = RULES[sn].replace("SV", sv)

                    missing_slots.append(sv)
            temp_dict["mr"] = mr

            best_ref = ref
            for missing_slot in missing_slots:
                if "friendly" in missing_slot:
                    best_ref = best_ref + " " + missing_slot
                else:
                    best_ref = self.insert_missingslot(mr, best_ref, missing_slot, t5_data_prep)

            temp_dict["ref"] = best_ref
            hill_climb_res.append(temp_dict)
        json.dump(hill_climb_res, open(outfile, 'w+'))

        print("Hill climb results dumped!!")
