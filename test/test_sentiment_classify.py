from opinion_mining.sentiment_classify import SentimentClassify as sc
from opinion_mining.sentiment_classify import SentimentClassifyChange as scc
from opinion_mining.text_extract import TextExtract as te
import time


def test_feature_extract():
    senti_df = te.read_sentences_from_sentiment_label()
    labeled_df = senti_df[senti_df.pos_neu_neg != -2]
    for index, row in labeled_df.iterrows():
        print '----------------------------------------------------'
        print sc.senti_feature_extract(row['review_id'], row['sentence_id'])


def test_get_train_set():
    print sc.get_final_train_set()


def test_train():
    sc.train()


def test_test():
    sc.test()


def test_train2():
    scc.train()


def test_test2():
    scc.test()


if __name__ == "__main__":
    # TODO encoding utf-8 in a more elegant way
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    test_train()
    time.sleep(1)
    test_test()