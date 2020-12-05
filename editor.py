from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

class RobertaEditor():
    def __init__(self, model_dir="/Users/jolly/PycharmProjects/e2e-dataset/roberta-ft-mlm/"):
        self.model_dir = model_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = RobertaForMaskedLM.from_pretrained(self.model_dir, return_dict=True)

    def generate(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt")
        # print([self.tokenizer.decode(i) for i in inputs["input_ids"][0]])
        mask_idx = \
        [i for i, j in enumerate([self.tokenizer.decode(i) for i in inputs["input_ids"][0]]) if j == "<mask>"][0]
        outputs = self.model(**inputs)
        return input_texts.replace(" <mask>", self.tokenizer.decode(torch.argmax(outputs.logits[0, mask_idx, :])))

    def insert(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx:]
        return self.generate(" ".join(input_texts_with_mask_list))

    def replace(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx + 1:]
        return self.generate(" ".join(input_texts_with_mask_list))

    def delete(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def get_contextual_word_embeddings(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].squeeze()[1:-1, :]