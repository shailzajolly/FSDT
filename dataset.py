from nlp import Dataset as NlpDataset
import json
from torch.utils.data import Dataset 


class D2tDataset(Dataset):
    def __init__(self, tokenizer, filepath, data_split, num_samples, input_length, output_length, print_text=False):
        self.dataset = self.read_dataset(filepath)[data_split]
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def read_dataset(self, filepath):

        dataset = {}
        raw_data = json.load(open(filepath))

        for split in ['train', 'validation', 'test']:
            split_dict = {"mr": [], "ref": []}
            for idx, i in enumerate(raw_data[split]):
                split_dict["mr"].append(i["mr"])
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
