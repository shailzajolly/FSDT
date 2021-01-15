import json

def prepare_data_t5_train(mrs, refs):

    e2e_like_list = []     #to run for T5 model

    for mr, ref in zip(mrs, refs):

        temp_dict = {}

        mr = mr.strip()
        ref = ref.strip()
        e2e_mr = {}
        for item in mr.split("\t"):

            sn, sv = item.split(":")[0], item.split(":")[1]
            sn = sn.split("_")
            if "<none>" in sv or "image" in sn or "px" in sv:
                continue
            if len(sn) == 1:
                e2e_sn = sn[0]
            else:
                e2e_sn = " ".join(sn[:-1])

            if e2e_sn not in e2e_mr:
                e2e_mr[e2e_sn] = []

            e2e_mr[e2e_sn].append(sv)
        e2e_mr = {k: " ".join(v) for k,v in e2e_mr.items()}
        e2e_mr = ", ".join([f"{k}[{v}]" for k, v in e2e_mr.items()])
        print(e2e_mr)
        print(ref)
        print("----------")

        temp_dict["mr"] = e2e_mr
        temp_dict["ref"] = ref
        e2e_like_list.append(temp_dict)

    return e2e_like_list

if __name__=='__main__':

    DATAPATH = '/Users/jolly/PycharmProjects/e2e-dataset/wikibio_T5/'
    PWD = '/Users/jolly/PycharmProjects/Few-Shot-NLG/data_release/humans/original_data/'
    PWD_ORG_DATA = '/Users/jolly/PycharmProjects/Few-Shot-NLG/data_release/original/'

    e2e_like_dict = {}

    data_source_train = PWD_ORG_DATA + 'train_500.box'
    text_source_train = PWD_ORG_DATA + 'train_500.summary'
    mrs = [line for line in open(data_source_train, 'r')]
    refs = [line for line in open(text_source_train, 'r')]
    e2e_like_list_train= prepare_data_t5_train(mrs, refs)
    e2e_like_dict["train"] = e2e_like_list_train

    data_source_valid = PWD + 'valid.box'
    text_source_valid = PWD + 'valid.summary'
    mrs = [line for line in open(data_source_valid, 'r')]
    refs = [line for line in open(text_source_valid, 'r')]
    e2e_like_list_valid = prepare_data_t5_train(mrs, refs)
    e2e_like_dict["validation"] = e2e_like_list_valid

    data_source_test = PWD + 'test.box'
    text_source_test = PWD + 'test.summary'
    mrs = [line for line in open(data_source_test, 'r')]
    refs = [line for line in open(text_source_test, 'r')]
    e2e_like_list_test = prepare_data_t5_train(mrs, refs)
    e2e_like_dict["test"] = e2e_like_list_test

    #json.dump(e2e_like_dict, open(DATAPATH + 'wikibio_500train_noNone.json','w+'))

