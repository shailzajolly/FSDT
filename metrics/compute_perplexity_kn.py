from io import open

import nltk
from nltk.lm import MLE
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.lm import Laplace, KneserNeyInterpolated

import csv
'''
-> 5-Gram Kneser-Ney perplexity
'''

FOLDER_PATH = 't5_ckpts/t5_small/e2e_100par/'#'../t5_ckpts/t5_small/e2e_1par_4psd_sa/'
FILE_PATH = 'cleaned_100par.txt'

FOLDER_PATH = 'other_baselines/'#'../t5_ckpts/t5_small/e2e_1par_4psd_sa/'
FILE_PATH = 'tgen.txt'

FOLDER_PATH = 't5_ckpts/t5_small/e2e_1par_4psd_hc/'#'../t5_ckpts/t5_small/e2e_1par_4psd_sa/'
FILE_PATH = 'tuned2.txt'

FOLDER_PATH = 'E2E/'#
FILE_PATH = 'e2e_100par.txt'

TRG_PATH = 'e2edataset/'
TRG_FILE = 'testset_w_refs.csv'

# real calculations


#open target set
corpus = []
with open(TRG_PATH+TRG_FILE,'r',encoding='utf-8') as f:
    reader = csv.reader(f, delimiter = ',')

    #lines = f.readlines()
    next(reader)
    for line in reader:
        corpus.append(line[1])

tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                for sent in corpus]

# kn5

n = 5
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
lm = KneserNeyInterpolated(n)

lm.fit(train_data,padded_vocab)

#src
src = []
with open(FOLDER_PATH+FILE_PATH,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        src.append(line[:-1])


tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                for sent in src]
test_data, padded_voc = padded_everygram_pipeline(n,tokenized_text)
#lm2 = Laplace(n)
#lm2.fit(test_data,padded_voc)

# for test in test_data:
#     print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),lm.score(ngram[-1], ngram[:-1])) for ngram in test])
score = 0
count = 0

for i, test in enumerate(test_data):
    if i%100 == 0:
        print(i)
    ppl = lm.perplexity(test)
    score += ppl

    #print("PP({0}):{1}".format(src[i], ppl))
print("i",i)
print(count)
print(score)
print(FILE_PATH, score/(i+1))


