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

SLOTNAMES_GLOB_wb = {'fullname', 'birth date', 'currentclub', 'nationality', 'occupation', 'position', 'death date', 'party', 'birth place'}
RULESWB = {'fullname': "SV",
         'birth date': "born on SV", #born on
         'currentclub': "plays for SV", #plays for
         'nationality': "SV",
         'occupation': "is a SV", #is a
         'position': "SV",
         'death date 1': "died on SV", #died on
         'death date 2': "died in SV", #died on
         'party': "serving in SV party", #serving in SV party
         'birth place': "born in SV"}#born in


class SimulatedAnnealing:

    def __init__(self, editor, generator, t_init, C, fluency_weight, semantic_weight, max_steps):

        self.editor = editor
        self.generator = generator
        self.t_init = t_init
        self.C = C
        self.fluency_weight = fluency_weight
        self.semantic_weight = semantic_weight
        self.max_steps = max_steps
        self.t5_data_prep = T5ScorerDataset(self.generator.tokenizer, generator.hparams.max_input_length,
                                            generator.hparams.max_output_length)

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
        self.input_length = model.hparams.max_input_length
        self.output_length = model.hparams.max_output_length

    def hasNumbers(self, inputString):
        return bool(re.search(r'\d', inputString))

    def insert_missingslot(self, mr, ref, missing_slot, t5_data_prep, batch_size=20):
        hypothesis = []
        hypothesis_scores = []
        tokens = ref.split()
        for i in range(len(tokens)):
            hypothesis.append(" ".join(tokens[:i] + [missing_slot] + tokens[i:]))

        for start_idx in range(0, len(hypothesis), batch_size):

            hypo_batch = hypothesis[start_idx:start_idx+batch_size]
            batch = t5_data_prep.get_batch(mr, hypo_batch)
            outs = self.model.fluency_score(batch)
            hypothesis_scores += outs.cpu().detach().numpy().tolist()

        #return hypothesis[np.random.choice(np.argsort(hypothesis_scores)[-3:], 1)[0]] #For WikiBio we did to have random sampling in HC
        return hypothesis[np.argsort(hypothesis_scores)[-1]] #For e2e exps

    def create_hillclimbing_data(self, mr_file, psd_ref_file, num_samples):
        
        #mr_refs = json.load(open(mr_file, 'r'))["train"][num_samples:]
        mr_refs = json.load(open(mr_file, 'r'))["test"] #for search in inference
        psd_refs = open(psd_ref_file, 'r')

        mr_psd_ref = []
        for mr_ref, psd_ref in zip(mr_refs, psd_refs):
            temp_mr = mr_ref["mr"].lower()
            temp_psd_ref = psd_ref.strip().lower()
            mr_psd_ref.append([temp_mr, temp_psd_ref])

        return mr_psd_ref

    def adding_missing_slotvalues(self,mr_file, ref_file, outfile, dataset_name, num_samples):

        if dataset_name == "e2e":
            self.adding_missing_slotvalues_e2e(mr_file, ref_file, outfile, num_samples)
        elif dataset_name == "wikibio":
            self.adding_missing_slotvalues_wb(mr_file, ref_file, outfile, num_samples)

    def adding_missing_slotvalues_e2e(self, mr_file, ref_file, outfile, num_samples=420):

        search_inference = open(outfile, 'w+')
        mr_pseudoref = self.create_hillclimbing_data(mr_file, ref_file, num_samples)
        t5_data_prep = T5ScorerDataset(self.tokenizer, self.input_length, self.output_length)
        hill_climb_res = []

        for mr_ref in tqdm.tqdm(mr_pseudoref):
            temp_dict = {}

            mr = mr_ref[0].strip().lower()
            ref = mr_ref[1].strip().strip(".").lower()

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
                            sv = RULES["familyfriendly"].strip().replace("SV", "")
                        #sv  = "it is " + sv.replace("family", random.sample(FF_TYPES, 1)[0]) +  "." #boolean constraint
                    else:
                        sv = RULES[sn].replace("SV", sv)

                    missing_slots.append(sv)
            temp_dict["mr"] = mr

            best_ref = ref
            for missing_slot in missing_slots:
                #if "friendly" in missing_slot:
                    #best_ref = best_ref + " " + missing_slot
                #else:
                best_ref = self.insert_missingslot(mr, best_ref, missing_slot, t5_data_prep, batch_size=64)

            temp_dict["ref"] = best_ref
            hill_climb_res.append(temp_dict)
            search_inference.write(best_ref+"\n")

        #json.dump(hill_climb_res, open(outfile, 'w+'))
        search_inference.close()

        print("Hill climb results written!")

    def adding_missing_slotvalues_wb(self, mr_file, ref_file, outfile, num_samples=100):

        search_inference = open(outfile, 'w+')
        mr_pseudoref = self.create_hillclimbing_data(mr_file, ref_file, num_samples)
        t5_data_prep = T5ScorerDataset(self.tokenizer, self.input_length, self.output_length)

        hill_climb_res = []

        for mr_ref in tqdm.tqdm(mr_pseudoref):

            temp_dict = {}

            mr = mr_ref[0].strip().lower()
            ref = mr_ref[1].strip().strip(".").lower() + " ."

            slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
            slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [

            slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]
            slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

            missing_slots = []
            coverage = 0
            occupation_counter = 0
            occupation_vals = []

            position_counter = 0
            position_vals = []

            name_counter = 0
            name_vals = []

            bd_counter = 0
            bd_vals = []

            bp_counter = 0
            bp_vals = []

            dd_counter = 0
            dd_vals = []

            for sn, sv in zip(slotnames, slotvalues):

                if sv in ref:
                    coverage += 1
                else:
                    if sn in SLOTNAMES_GLOB_wb:

                        if sn == "occupation":
                            if "," in sv:
                                split_svs = map(str.strip, sv.split(","))
                            else:
                                split_svs = map(str.strip, sv.split(" "))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    occupation_counter+=1
                                else:
                                    occupation_vals.append(split_sv)

                            if occupation_counter == 0:
                                sv = RULESWB[sn].replace("SV", " ".join(occupation_vals))
                                missing_slots.append(sv)
                        
                        elif sn == "birth place":
                            split_svs = map(str.strip, sv.split(","))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    bp_counter+=1
                                else:
                                    bp_vals.append(split_sv)

                            if bp_counter == 0:
                                sv = RULESWB[sn].replace("SV", " ".join(bp_vals))
                                missing_slots.append(sv)
                        
                        elif sn == "position":
                            if "/" in sv:
                                split_svs = map(str.strip, sv.split("/"))
                            else:
                                split_svs = map(str.strip, sv.split(" "))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    position_counter+=1
                                else:
                                    position_vals.append(split_sv)

                            if position_counter == 0:
                                sv = RULESWB[sn].replace("SV", " ".join(position_vals))
                                missing_slots.append(sv)

                        elif sn == "fullname":
                            split_svs = map(str.strip, sv.split(" "))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    name_counter+=1
                                else:
                                    name_vals.append(split_sv)

                            if name_counter == 0:
                                sv = RULESWB[sn].replace("SV", " ".join(name_vals))
                                missing_slots.append(sv)

                        elif sn == "birth date":
                            split_svs = map(str.strip, sv.split(" "))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    bd_counter+=1
                                else:
                                    bd_vals.append(split_sv)

                            if bd_counter == 0:
                                sv = RULESWB[sn].replace("SV", " ".join(bd_vals))
                                missing_slots.append(sv)

                        elif sn == "death date":
                            split_svs = list(map(str.strip, sv.split(" ")))
                            for split_sv in split_svs:
                                if split_sv in ref:
                                    dd_counter+=1
                                else:
                                    dd_vals.append(split_sv)

                            if dd_counter == 0:
                                if len(split_svs) > 2:
                                    sv = RULESWB[sn+" 1"].replace("SV", " ".join(dd_vals))
                                else:
                                    sv = RULESWB[sn+" 2"].replace("SV", " ".join(dd_vals))
                                missing_slots.append(sv)

                        else:
                            sv = RULESWB[sn].replace("SV", sv)
                            missing_slots.append(sv)
        

            temp_dict["mr"] = mr
            best_ref = ref

            for missing_slot in missing_slots:
                best_ref = self.insert_missingslot(mr, best_ref, missing_slot, t5_data_prep, 10)

            if False and len(missing_slots)!=0:
                print(mr)
                print(ref)
                print(missing_slots)
                print("best_ref:", best_ref)
                print("-----------------------")
                input()


            temp_dict["ref"] = best_ref
            hill_climb_res.append(temp_dict)
            search_inference.write(best_ref+"\n")

        #json.dump(hill_climb_res, open(outfile, 'w+'))
        search_inference.close()

        print("Hill climb results written!")
