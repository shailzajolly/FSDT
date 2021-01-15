import pandas
import re
import json

'''
Parent script accepts following formats:- 
For tables: pairs of attributes and values: attribute_1|||value_1<TAB>attribute_2|||value_2<TAB>... The <table_file> should contain the GT tables corresponding to these in each line.
For references: Multiple references should be separated by <TAB>s on the same line.
'''

def write_tabref_file_wikibio():

    WB_DATAPATH = '/Users/jolly/PycharmProjects/e2e-dataset/wikibio_T5/wikibio_100train_noNone.json'

    tab_test = open('tab_test_wb.txt','w')
    ref_test = open('ref_test_wb.txt','w')

    test_data = json.load(open(WB_DATAPATH, 'r'))

    MR_Te = test_data["test"]

    for mrref in MR_Te:

        mr = mrref["mr"].lower()
        ref = mrref["ref"].lower()

        temp_ls = []

        slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
        slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]

        slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [
        slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

        for sn, sv in zip(slotnames, slotvalues):

            temp_ls.append(sn + "|||" + sv)

        tab_test.write("\t".join(temp_ls) + "\n")
        ref_test.write(ref + "\n")

def write_tab_file_e2e():

    tab_test = open('tab_test.txt','w')

    colnames = ["mr", "ref"]
    test_data = pandas.read_csv(DATAPATH + 'e2e-dataset/testset.csv', names=colnames)

    MR_Te = test_data.mr.tolist()
    for mr in MR_Te[1:]:

        mr = mr.strip().lower()
        temp_ls = []

        slotvalues_reg = re.finditer(r'\[.*?\]', mr)  # takes out values between []
        slotvalues = [v.group(0).strip('[]') for v in slotvalues_reg]

        slotnames_reg = re.finditer(r'\w+\s*\w*?\[', mr)  # takes out names before [
        slotnames = [v.group(0).strip('[]') for v in slotnames_reg]

        for sn, sv in zip(slotnames, slotvalues):
            if "friendly" in sn and sv in ["yes"]:
                sv = "friendly friendly"

            elif "friendly" in sn and sv in ["no", "not"]:
                sv = "not friendly friendly"
            temp_ls.append(sn + "|||" + sv)

        tab_test.write("\t".join(temp_ls) + "\n")


def write_reference_file_e2e(test_mr_refs):

    ref_test = open('ref_test.txt', 'w')

    MR_Te = test_mr_refs.mr.tolist()
    REF_Te = test_mr_refs.ref.tolist()

    counter = 1
    temp_mr = [MR_Te[1]]
    temp_refs = []

    for mean_rep, mean_ref in zip(MR_Te[1:], REF_Te[1:]):
        mean_ref = mean_ref.strip().lower()

        if mean_rep in temp_mr:
            temp_refs.append(mean_ref)
            temp_mr = []
            temp_mr.append(mean_rep)

        else:
            ref_test.write("\t".join(temp_refs) + "\n")
            temp_refs = []
            temp_mr = []
            temp_mr.append(mean_rep)
            temp_refs.append(mean_ref)

    ref_test.close()
    print("DONE!!!")


if __name__=='__main__':


    DATAPATH = '/Users/jolly/PycharmProjects/e2e-dataset/'

    #colnames = ["mr", "ref"]
    #test_mr_refs = pandas.read_csv(DATAPATH + 'e2e-dataset/testset_w_refs.csv', names=colnames)
    #write_reference_file_e2e(test_mr_refs)

    #write_tabref_file_wikibio()