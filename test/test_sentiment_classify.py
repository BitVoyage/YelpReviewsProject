from opinion_mining.sentiment_classify import SentimentClassify as sc
from opinion_mining.text_extract import  TextExtract as te


def test_feature_extract():
    senti_df = te.read_sentences_from_sentiment_label()
    labeled_df = senti_df[senti_df.pos_neu_neg != -2]
    for index, row in labeled_df.iterrows():
        print '----------------------------------------------------'
        print sc.senti_feature_extract(row['review_id'], row['sentence_id'])


if __name__ == "__main__":
    # TODO encoding utf-8 in a more elegant way
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    test_feature_extract()
