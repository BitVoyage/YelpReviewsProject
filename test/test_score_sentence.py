from opinion_mining.sentence_score import ScoreBasedOnSentiWordNet as sc
from opinion_mining.sentence_score import ScoreBasedOnAFINN as sba
from opinion_mining.sentence_score import ScoreBasedOnBingLiu as sbbl
from nltk import word_tokenize

PATH = '/home/zls/Projects/yelp_dataset/'
train_set_filename = 'sentiment_train_set.csv'
label_sentiment_filename = 'label_sentiment.txt'
label_aspects_filename = 'label_aspects.txt'


def get_sent_list():
    with open(PATH+label_aspects_filename, 'r') as f:
        sent_list = [sent.split('||')[1] for sent in f.readlines(20)]
    return sent_list


def test_sentiwordnet(sent_list):

    # def score_reduce(a, b):
    #     if not isinstance(a, list):
    #         a = []
    #     a.extend(b)
    #     return a

    # score_list = reduce(score_reduce, map(sc.get_score_pair_sentiwordnet, sent_list), [])
    # print score_list

    for sent in sent_list:
        print sent
        print sc.get_score(sent)


def test_sentiwordnet_simple_neg(sent_list):
    for sent in sent_list:
        print sent
        print sc.get_score_simple_neg(sent)


def test_neg_sent():
    sent_list = ["I don't really like the food.", "The sevice is not bad as reviews described"]
    for sent in sent_list:
        print sent
        print sc.get_score_simple_neg(sent)


def test_affin(sent_list):
    for sent in sent_list:
        print sent
        print sba.predict(word_tokenize(sent))


def test_bingliu(sent_list):
    for sent in sent_list:
        print sent
        print sbbl.predict(word_tokenize(sent))


def main():
    test_neg_sent()

if __name__ == "__main__":
    # TODO encoding utf-8 in a more elegant way
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
