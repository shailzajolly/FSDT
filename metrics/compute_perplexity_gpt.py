from tqdm import tqdm
import torch
import math
import numpy as np
import os
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from torch.nn import functional as F

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = 'cuda'
model_id = 'gpt2'  #'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
#sandl/t5_ckpts/t5_small/e2e_1par_4psd_hc/sa_test.txt
FOLDER_PATH = 'wikibio_T5_ckpts/wikibio_1par_4psd_hc_xtra/'#'../t5_ckpts/t5_small/e2e_1par_4psd_sa/'
FILE_PATH = '1par_4psd_hc.txt'

RES_FOLDER = 'ppl_result/'


dataset = load_dataset('text', data_files={'test': FOLDER_PATH + FILE_PATH}, split='test')
#to load particular test file over which perplexity will be computed

#to get scores sentence by sentencne
if not os.path.exists(RES_FOLDER):
	os.mkdir(RES_FOLDER)

perplexity_scores = open(RES_FOLDER + FILE_PATH,'w')

def score(sentence):
    #sentence = "blue spice is a pub serving chinese food i in the riverside area near family friendly rainbow vegetarian café"
    tokenize_input = tokenizer.tokenize(sentence, return_tensors='pt')
    # print(tokenize_input)
    
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
   
    outputs=model(tensor_input, labels=tensor_input)
    
    loss = outputs[0]
    logits = outputs[1]
  
    return math.exp(loss.item())#/len(tokenize_input))


ppls = []
#to get scores sentence by sentence
for data in tqdm(dataset):
    ppl = score(data['text'])
    #print('\n'+str(ppl))
    # exit()
    output = {"ref": data['text'], "ppl" : ppl}
    ppls.append(ppl) 
    perplexity_scores.write(json.dumps(output) + "\n")

print("For file: ", FILE_PATH)
print("PPL score:", np.mean(ppls))

perplexity_scores.close()

 #For Debugging

 	# print([tokenizer.decode(idx) for idx in tensor_input.cpu()[0].tolist()])
    #print(tensor_input.shape)
    #print(logits.shape) #(1, 11, 50257)
    #preds = -torch.gather(input=F.softmax(logits[:, :-1,:], dim=2), index=tensor_input[:,1:].unsqueeze(2), dim=2).squeeze().log()
    #print(preds)
    #print(preds.mean().exp())
    #print("ppl", math.exp(loss.item()))
    #input()    




