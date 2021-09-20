# %%
import json
import numpy as np
import pandas
import re
from io import open
import math
import string
# from sentence_transformers import SentenceTransformer

from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder
'''Objective
	-> Soft alignment between values, namely slotvalues and 
'''
scorer = CrossEncoder('cross-encoder/stsb-distilroberta-base')
def compute_coverage(MR, preds,model_preds):
	
	print("MR Length: ", len(MR))
	print("Preds Length: ", len(preds))
	pun = string.punctuation.replace('[','').replace(']','')
	
	coverage = 0
	total_coverage = []
	total_slots = 0
	boarderline = math.cos(45)

	maybe = []
	for mr, pred, model_pred in tqdm(zip(MR, preds,model_preds),total = len(MR)):
		

		mr = mr.lower()
		pred = pred.lower()
		for c in pred:
			if c in pun:
				pred = pred.replace(c,' ')

		for c in mr:
			if c in pun:
				mr = mr.replace(c,' ')
		
		model_pred = model_pred.lower()
		for c in model_pred:
			if c in pun:
				model_pred = model_pred.replace(c,' ')

		slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
		slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]
		
		pred = ' '.join(pred.split()) 
		
		model_pred = ' '.join(model_pred.split()) 
		temp = []
		slots = []
		for slotvalue in slotvalues:
			slotvalue=' '.join(slotvalue.split()) 

			if slotvalue in pred:
				slots.append(slotvalue)
				
			elif len(slotvalue.split()) >1:
				length = 0
				for token in slotvalue.split():
					if token in pred:
						length = 0 
					else:
						break
				if length ==len(slotvalue.split()):
					
					slots.append(slotvalue)
			
			else:
				temp.append([slotvalue,pred])
		
		if len(temp)>0:
			score = scorer.predict(temp)
		
			for idx, s in enumerate(score):
				if s>=boarderline: slots.append(temp[idx][0])

		for slotvalue in slots:
		

			
			if slotvalue in model_pred:
				coverage += 1
				
				
			elif len(slotvalue.split()) >1:
				length = 0
				for token in slotvalue.split():
					if token in model_pred:
						length = 0 
					else:
						break
				if length ==len(slotvalue.split()):
					coverage+=1
					
			
			else:
				maybe.append([slotvalue,model_pred])
		
		total_slots += len(slots)
	
		
	score = scorer.predict(maybe)

	coverage+=sum(x >=boarderline for x in score)
	
	return coverage/total_slots

if __name__=='__main__':
	
	PWD = 'wikibio_T5_ckpts/wikibio_1par_4psd_hc_xtra/'
	FILE = '1par_4psd_hc.txt'


	
	
	model_output = [pred.strip().lower() for pred in open( PWD + FILE, 'r',encoding='utf-8')]
	
	test_gt = 'data/test.csv'

	colnames = ["mr","ref"]
	
	test_data = pandas.read_csv(test_gt, names=colnames,delimiter=',')#,encoding='utf-8')
	
	test_data.mr = test_data.mr.str.lower()
	test_data.ref = test_data.ref.str.lower()
	MR_Te = test_data.mr.tolist()
	REF_Te = test_data.ref.tolist()
	
	coverage = compute_coverage(MR_Te[1:], REF_Te[1:], model_output)
	print("For file:",PWD+FILE)
	print("Coverage is: ", coverage)
	
	
	#[ashley corker] -> ashley james corker
# %%
