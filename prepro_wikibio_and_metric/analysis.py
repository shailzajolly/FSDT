import json
import numpy as np
import pandas
import re
from nltk.translate.bleu_score import sentence_bleu


SLOTNAMES_GLOB_wb = {'fullname', 'birth date', 'currentclub', 'nationality', 'occupation', 'position', 'death date', 'party', 'birth place'}

def compute_average_lens(sentences):

    lens = []

    '''
    test_gt = 'e2e-dataset/testset_w_refs.csv'
    colnames = ["mr", "ref"]
    test_data = pandas.read_csv(PWD + test_gt, names=colnames)
    MR_Te = test_data.mr.tolist()
    REF_Te = test_data.ref.tolist()
    
    for g in REF_Te:
        lens.append(len(g.strip().split(" ")))

    print(np.mean(lens))
    '''

    for sent in sentences:
        lens.append(len(sent.strip().split(" ")))
    print(np.mean(lens))

def extract_slotnames4HC(data, self_train_samples):

    slotnames_4_cov = {}
    coverage = 0
    cc = 0

    for mr_ref, st_samp in zip(data, self_train_samples):

        mr = mr_ref["mr"]
        ref = mr_ref["ref"]

        slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [
        slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

        slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
        slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]

        for slotname, slotvalue in zip(slotnames, slotvalues):

            if slotvalue in ref:
                coverage += 1
            else:

                if slotname in SLOTNAMES_GLOB_wb:
                    if slotvalue not in ref:
                        cc +=1
                        print(mr)
                        print(ref)
                        print(slotname)
                        print(slotvalue)
                        print("-------")




    #         if slotname not in slotnames_4_cov:
    #             slotnames_4_cov[slotname] = [0, 0]
    #
    #         if slotvalue in st_samp:
    #             slotnames_4_cov[slotname][0] += 1
    #
    #         slotnames_4_cov[slotname][1] += 1
    #
    # for slotname, counts in slotnames_4_cov.items():
    #     slotnames_4_cov[slotname] = [(counts[0]/counts[1]) * 100] + slotnames_4_cov[slotname]
    #
    # valid_slotnames_cov = [(k, slotnames_4_cov[k]) for k in sorted(slotnames_4_cov, key=slotnames_4_cov.get, reverse=True) if slotnames_4_cov[k][-1] >= 10 and slotnames_4_cov[k][0] >= 30]
    # print(valid_slotnames_cov)
    # print(len(valid_slotnames_cov))
    # print(len(data))
    #

def check_slotcoverage_wb(data):

    slotnames_4_cov = {}

    for mr_ref in data:

        mr = mr_ref["mr"]
        ref = mr_ref["ref"]

        slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [
        slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

        slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
        slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]

        for slotname, slotvalue in zip(slotnames, slotvalues):
            if slotname not in slotnames_4_cov:
                slotnames_4_cov[slotname] = [0,0]

            if slotvalue in ref:
                slotnames_4_cov[slotname][0]+=1

            slotnames_4_cov[slotname][1]+=1

    for slotname, counts in slotnames_4_cov.items():
        slotnames_4_cov[slotname] = [(counts[0]/counts[1]) * 100] + slotnames_4_cov[slotname]

    valid_slotnames_cov = [(k, slotnames_4_cov[k]) for k in sorted(slotnames_4_cov, key=slotnames_4_cov.get, reverse=True) if slotnames_4_cov[k][-1] >= 10 and slotnames_4_cov[k][0] >= 10]
    print(valid_slotnames_cov)
    print(len(valid_slotnames_cov))
    print(len(data))


def compare_2_files(samps1, samps2):

    mrs_score = []
    refs_score = []

    for samp1, samp2 in zip(samps1, samps2):

        mrs_score.append(sentence_bleu([samp1["mr"]],samp2["mr"]))
        refs_score.append(sentence_bleu([samp1["ref"]],samp2["ref"]))

    print(set(mrs_score))
    print(set(refs_score))


if __name__=='__main__':

    PWD = '/Users/jolly/PycharmProjects/e2e-dataset/'
    #model_output = [pred.strip().lower() for pred in open(PWD + 'e2e-metrics/example-inputs/LatestCode_t5_e2epreds/t5_small/e2e_1par_4psd_hc_boolconstraint.txt', 'r')]

    #wikibio_file_gt = [mrref["ref"] for mrref in json.load(open(PWD + 'wikibio_t5/wikibio_100train_noNone.json', 'r'))["test"]]
    #wikibio_file_preds = [sent for sent in open(PWD + 'e2e-metrics/example-inputs/wikibio/T5preds_wikibio_human/wikibio_100samples_noNone_T5.txt', 'r')]

    #compute_average_lens(wikibio_file_preds)

    data_pth_100 = json.load(open(PWD + 'wikibio_T5/wikibio_100train_noNone.json', 'r'))["train"]
    data_pth_500 = json.load(open(PWD + 'wikibio_T5/wikibio_500train_noNone.json', 'r'))["train"]

    psd_file = [sent for sent in open(PWD + 'e2e-metrics/example-inputs/wikibio/T5preds_wikibio_human/pseudopar400_genfrom100ckpt.txt', 'r')]
    print(len(psd_file))

    #check_slotcoverage_wb(data_pth_100)

    extract_slotnames4HC(data_pth_500[100:], psd_file)




    # 400
    # [('fullname', [79.16666666666666, 19, 24]), ('name', [71.42857142857143, 70, 98]),
    #  ('article title', [60.0, 60, 100]), ('birth date', [56.043956043956044, 51, 91]),
    #  ('currentclub', [53.84615384615385, 7, 13]), ('nationality', [47.61904761904761, 10, 21]),
    #  ('occupation', [46.15384615384615, 12, 26]), ('birth name', [45.45454545454545, 5, 11]),
    #  ('position', [40.0, 12, 30]), ('death date', [37.5, 9, 24]), ('party', [30.76923076923077, 4, 13]),
    #  ('birth place', [18.51851851851852, 15, 81]), ('goals', [10.0, 2, 20])]
    # 13
    # 100

    # [('fullname', [77.22772277227723, 78, 101]), ('nationality', [73.4375, 47, 64]), ('birth name', [68.75, 44, 64]),
    #  ('name', [67.98941798941799, 257, 378]), ('article title', [59.25, 237, 400]),
    #  ('birth date', [48.882681564245814, 175, 358]), ('position', [42.758620689655174, 62, 145]),
    #  ('death date', [40.909090909090914, 45, 110]), ('occupation', [40.18691588785047, 43, 107])]