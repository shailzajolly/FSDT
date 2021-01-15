from nlp import Dataset as NlpDataset
import json
from torch.utils.data import Dataset

import torch


class D2tDataset(Dataset):
    def __init__(self, tokenizer, filepath, augment_data_filepath, data_variant, data_split, data_variant_samples,
                 num_samples, input_length, output_length, print_text=False):

        self.dataset = self.read_dataset(filepath, augment_data_filepath, data_variant, data_variant_samples)[data_split]
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        print("Split: ",data_split)
        print("Stats: ", self.dataset)

        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def read_dataset(self, filepath, augment_data_filepath, data_variant, data_variant_samples):

        dataset = {}
        raw_data = json.load(open(filepath))


        #For different variants of data
        if data_variant!="":
            print(f"Data variant: {data_variant} | Number of samples: {data_variant_samples}")

        augment_data = None
        if data_variant=="gen_psd_4par":
            raw_data["test"] = raw_data["train"][data_variant_samples:]
        elif data_variant=="1par_4psd":
            augment_data = [line.strip() for line in open(augment_data_filepath, 'r')]
        elif data_variant=="1par_4psd_hc":
            augment_data = json.load(open(augment_data_filepath, 'r'))
            raw_data['train'] = raw_data['train'][:data_variant_samples] + augment_data

        for split in ['train', 'validation', 'test']:
            split_dict = {"mr": [], "ref": []}
            for idx, i in enumerate(raw_data[split]):

                #To add pseudo-ref for 4 par
                if data_variant=="1par_4psd" and augment_data:
                    if split=="train" and idx >= data_variant_samples:
                        i["ref"] = augment_data[idx-data_variant_samples]

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
