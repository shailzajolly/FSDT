# %%
import json
import numpy as np
import pandas
import re
from io import open
import math
import string
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder
from collections import Counter,OrderedDict
from itertools import islice

'''Objective
	-> Soft alignment between values, namely slotvalues and 
'''
import matplotlib.pyplot as plt
scorer = CrossEncoder('cross-encoder/stsb-distilroberta-base')


def compute_coverage(MR, preds):
	# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
	print("MR Length: ", len(MR))
	print("Preds Length: ", len(preds))
	pun = string.punctuation.replace('[','').replace(']','')
	
	coverage = 0
	total_coverage = []
	
	boarderline = math.cos(45)
	maybe = []
	new = []
	total_slots = 0
	
	slot_names = []

	for mr, pred in tqdm(zip(MR, preds),total = len(MR)):
		

		mr = mr.lower()
		pred = pred.lower()
		for c in pred:
			if c in pun:
				pred = pred.replace(c,' ')
		for c in mr:
			if c in pun:
				mr = mr.replace(c,' ')
			
		slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
		slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]
		
		slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [
		slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

		pred = ' '.join(pred.split()) 
		total_slots+=len(slotvalues)
		for i,slotvalue in enumerate(slotvalues):
			
			slotvalue=' '.join(slotvalue.split()) 
			

			if slotvalue in pred:
				coverage += 1
				
				
				slot_names.append(slotnames[i])
			elif len(slotvalue.split()) >1:
				length = 0
				for token in slotvalue.split():
					if token in pred:
						length += 1
					else:
						break
				if length ==len(slotvalue.split()):
					coverage+=1
					
					slot_names.append(slotnames[i])
			else:
				
				maybe.append([slotvalue,pred])

		
		
		
		cov=0

	
	print(coverage/total_slots)
	print('coverage:',coverage)
	
	score = scorer.predict(maybe)
	
	
	coverage+=sum(x >=boarderline for x in score)
	
	
	return coverage/total_slots

if __name__=='__main__':
	
	PWD = 'wikibio_T5_ckpts/wikibio_1par_4psd_hc_xtra/'# wikibio_1par_4psd_hc_predict/'#'test1/test_outputs/E2E/'
	
	
	FILE = '1par_4psd_hc.txt'#'1par_4psd_hc2.txt'
	
	
	model_output = [pred.strip().lower() for pred in open( PWD + FILE, 'r',encoding='utf-8')]
	
	test_gt = 'data/test.csv'

	colnames = ["mr","ref"]
	
	test_data = pandas.read_csv(test_gt, names=colnames,delimiter=',')#,encoding='utf-8')
	
	test_data.mr = test_data.mr.str.lower()
	MR_Te = test_data.mr.tolist()
	
	test_data.ref = test_data.ref.str.lower()
	REF_Te = test_data.ref.tolist()


	
	coverage = compute_coverage(MR_Te[1:], model_output)
	print("For file:",PWD+FILE)
	print("Coverage is: ", coverage)
	
	
	# [ashley corker] -> ashley james corker





# %%
