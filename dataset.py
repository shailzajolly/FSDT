from nlp import Dataset as NlpDataset
import json
from torch.utils.data import Dataset

import torch


class D2tDataset(Dataset):
    def __init__(self, tokenizer, filepath, data_split, num_samples, input_length, output_length, print_text=False):
        self.dataset = self.read_dataset(filepath)[data_split]
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        print("Split: ",data_split)
        print("Stats: ", self.dataset)

        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def read_dataset(self, filepath):

        dataset = {}
        raw_data = json.load(open(filepath))

        '''
        For different variants of data
        '''
        # raw_data1 = json.load(open('data/viggo_t5lower.json')) #wholeTrainingData
        # raw_data = raw_data1 #To train on whole Training data

        # raw_data["test"] = raw_data["train"][420:] #For generating pseudo samples for 4%
        # raw_data['test'] = raw_data['train'][:420] #For generating pseudo samples for same 1%

        raw_data['train'] = raw_data['train'][:420] #For 1% few-shot experiment
        # raw_data['test'] = raw_data1['train'] #For test on whole training data

        # For using 1% training samples + 4% Hill-climb results
        # hc_res = json.load(open('hill_climb_results_4perc.json','r'))
        # raw_data['train'] = raw_data['train'][:420] + hc_res

        # For using pseudo samples of 100% in training
        # pseudopar100 = [line.strip() for line in open('pseudopar100_30epochs_1par.txt','r')]
        # raw_data['train'] = raw_data['train'][:420] + raw_data1['train']

        ##For using pseudo samples of 4% in training
        # pseudopar4 = [line.strip() for line in open('pseudopar4_30epochs_1par.txt','r')]
        # raw_data['train'] = raw_data['train']

        ##For using same 1% pseudo samples in training
        #     pseudopar1 = [line.strip() for line in open('pseudopar1_30epochs_1par.txt','r')]
        #     raw_data['train'] = raw_data['train'][:420] + raw_data['train'][:420] + raw_data['train'][:420]

        ##For using same 1% pseudo samples in training + v2 results
        #     pseudopar1_v2 = [line.strip() for line in open('pseudopar1_30epochs_1parv2.txt','r')]

        for split in ['train', 'validation', 'test']:
            split_dict = {"mr": [], "ref": []}
            for idx, i in enumerate(raw_data[split]):
                # To add pseudo-ref for 100 par
                # if split=="train" and idx >= 420:
                #     i["ref"] = pseudopar100[idx-420]

                #To add pseudo-ref for 4 par
                # if split=="train" and idx >= 420:
                #     i["ref"] = pseudopar4[idx-420]

                # To add pseudo-ref for same 1 par
                # if split=="train" and 420<=idx<840:
                #     i["ref"] = pseudopar1[idx-420]

                # To add pseudo-ref for same 1 par v2
                # if split=="train" and idx>=840:
                #     i["ref"] = pseudopar1_v2[idx-840]

                split_dict["mr"].append(i["mr"])
                # split_dict["ref"].append(i["ref"]) #uncomment if next module not running
                if split=="test":
                    split_dict["ref"].append("ref")
                else:
                    split_dict["ref"].append(i["ref"])

            dataset[split] = NlpDataset.from_dict(split_dict)

        return dataset

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def convert_to_features(self, example_batch):
        # Tokenize mr and ref (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['mr']))

        input_ = self.clean_text(example_batch['mr'])
        target_ = self.clean_text(example_batch['ref'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}



class T5ScorerDataset(Dataset):
    def __init__(self, tokenizer, input_length, output_length):
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n' ,'')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text


    def convert_to_features(self, mr, ref):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(mr)
        target_ = self.clean_text(ref)

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return source, targets


    def _getitem(self, mr, ref):
        source, targets = self.convert_to_features(mr, ref)

        source_ids = source["input_ids"  ]  # .squeeze()
        target_ids = targets["input_ids"  ]  # .squeeze()

        src_mask    = source["attention_mask"  ]  # .squeeze()
        target_mask = targets["attention_mask"  ]  # .squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


    def get_batch(self, mrs, refs):
        batch = {"source_ids": [],
                 "source_mask": [],
                 "target_ids": [],
                 "target_mask": []}

        if len(mrs)==1:
            mrs = mrs * len(refs)

        for item in [self._getitem(mr, ref) for mr, ref in zip(mrs,refs)]:
            batch["source_ids"].append(item["source_ids"])
            batch["source_mask"].append(item["source_mask"])
            batch["target_ids"].append(item["target_ids"])
            batch["target_mask"].append(item["target_mask"])

        return {k: torch.cat(v, dim=0).cuda() for k, v in batch.items()}
