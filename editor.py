from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import numpy as np


class RobertaEditor():
    def __init__(self, model_dir="/Users/jolly/PycharmProjects/e2e-dataset/roberta-ft-mlm/"):
        self.model_dir = model_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = RobertaForMaskedLM.from_pretrained(self.model_dir, return_dict=True)
        self.ops_map = [self.insert, self.replace, self.delete]

    def edit(self, inputs, ops, positions):

        masked_inputs = np.array([self.ops_map[op](inp, position) for inp, op, position, in zip(inputs, ops, positions)])
        insert_and_replace_inputs = masked_inputs[np.where(ops<2)]
        insert_and_replace_outputs = self.generate(insert_and_replace_inputs)
        masked_inputs[np.where(ops < 2)] = insert_and_replace_outputs

        return masked_inputs


    def generate(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt")
        # print([self.tokenizer.decode(i) for i in inputs["input_ids"][0]])
        mask_idxs = self.get_mask_indexes(inputs)
        outputs = self.model(**inputs)
        mask_words = self.get_word_at_mask(self, outputs, mask_idxs).squeeze().cpu().numpy()

        return np.array([input_text.replace(" <mask>", mask_word) for input_text, mask_word in zip(input_texts, mask_words)])


    def insert(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx:]
        return " ".join(input_texts_with_mask_list)

    def replace(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def delete(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def get_mask_indexes(self, input_tensors):
        return [mask_idx for input_tensor in input_tensors
                for mask_idx, value in enumerate([self.tokenizer.decode(idx) for idx in input_tensor["input_ids"]])
                if value == "<mask>"]

    def get_word_at_mask(self, output_tensors, mask_idxs):

        mask_idxs = torch.Tensor(mask_idxs).long().unsqueeze(dim=1)
        return self.tokenizer.decode(torch.argmax(output_tensors.logits, dim=2).gather(1, mask_idxs))


    def get_contextual_word_embeddings(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].squeeze()[1:-1, :]

