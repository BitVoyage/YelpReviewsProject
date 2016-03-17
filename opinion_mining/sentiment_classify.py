# import pandas as pd
from sentence_score import ScoreBasedOnSentiWordNet as sbswn
from text_extract import TextExtract as te


class SentimentClassify:

    _label_sentiment = None
    _train_set = None

    def __init__(self):
        pass

    @classmethod
    def get_label_sentiment(cls):
        if cls._label_sentiment is None:
            cls._label_sentiment = te.read_sentences_from_sentiment_label()
        return cls._label_sentiment

    @classmethod
    def get_train_set(cls):
        if cls._train_set is None:
            cls._train_set = te.read_sentences_from_trainset()
        return cls._train_set

    @classmethod
    def get_score(cls, review_id, sentence_id):
        sentence_id = int(sentence_id)
        train_set = cls.get_train_set()
        row = train_set[(train_set.review_id == review_id) & (train_set.sentence_id == sentence_id)].iloc[0]
        return sbswn.get_score_simple_neg(row['sentence'])

    @classmethod
    def get_review_score(cls, review_id):
        train_set = cls.get_train_set()
        # get Series of sentences, then join sentences together
        sentences = train_set[train_set.review_id == review_id]['sentence'].str.cat()
        return sbswn.get_score_simple_neg(sentences)

    @classmethod
    def senti_feature_extract(cls, review_id, sentence_id):
        sentence_id = int(sentence_id)
        train_set = cls.get_train_set()
        row = train_set[
            (train_set.review_id == review_id) & (train_set.sentence_id == sentence_id)].iloc[0]
        # get this review's last sentence id
        max_sent_id = max(train_set[(train_set.review_id == row['review_id'])]['sentence_id'])

        # features: overall_stars, sent_i score, sent_i-1 score, sent_i+1 score, review score
        overall_stars = row['overall_stars']
        this_score = sbswn.get_score_simple_neg(row['sentence'])
        pre_sent_score = cls.get_score(review_id, max(sentence_id - 1, 0))
        next_sent_score = cls.get_score(review_id, min(sentence_id + 1, max_sent_id))
        review_score = cls.get_review_score(review_id)

        return [overall_stars, this_score, pre_sent_score, next_sent_score, review_score]






def main():

    print


if __name__ == "__main__":
    main()
