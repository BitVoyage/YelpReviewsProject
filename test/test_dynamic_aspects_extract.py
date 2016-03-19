from opinion_mining.dynamic_aspects_extract import DynamicAspectsExtract as dae
from opinion_mining.text_extract import TextExtract as te
from random import randint

def test_np_parse(sent):
    print dae().syntactic_parse(sent)


def test_leaves(sent):
    for a in dae().get_np(sent):
        print a


def test_get_candidate_aspects(sent):
    dae().get_candidate_aspects(sent)


def get_random_business_id():
    busi_ids = te.read_sentences_from_trainset().business_id
    i = randint(0, len(busi_ids))
    return busi_ids[i]

def test_get_aspects():
    for i in range(30):
        busi_id = get_random_business_id()
        dae(busi_id).get_aspects()

if __name__ == "__main__":
    # TODO encoding utf-8 in a more elegant way
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    # test_train()
    test_get_aspects()
