# import pandas as pd
from sentence_score import ScoreBasedOnSentiWordNet as sbswn
from text_extract import TextExtract as te
from nltk.classify import MaxentClassifier as maxent
import pickle


class SentimentClassify:

    LABELED_NUM = 300
    MAXENT_ALGORITHM = 'IIS'
    CLASSIFIER_FILE = '../data/classifier.pickle'
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
    def get_label(cls, review_id, sentence_id):
        label_sentiment = cls.get_label_sentiment()
        row = label_sentiment[(label_sentiment.review_id == review_id) & (label_sentiment.sentence_id == sentence_id)]
        return row.iloc[0]['pos_neu_neg']

    @classmethod
    def senti_feature_extract(cls, review_id, sentence_id):
        sentence_id = int(sentence_id)
        train_set = cls.get_train_set()
        row = train_set[
            (train_set.review_id == review_id) & (train_set.sentence_id == sentence_id)].iloc[0]
        # get this review's last sentence id
        max_sent_id = max(train_set[(train_set.review_id == row['review_id'])]['sentence_id'])

        # features: overall_stars, sent_i score, sent_i-1 score, sent_i+1 score, review score
        overall_stars = round(row['overall_stars']/5, 2)
        this_score = sbswn.get_score_simple_neg(row['sentence'])
        pre_score = cls.get_score(review_id, max(sentence_id - 1, 0))
        next_score = cls.get_score(review_id, min(sentence_id + 1, max_sent_id))
        review_score = cls.get_review_score(review_id)

        return dict(overall_stars=overall_stars, this_score=this_score, pre_score=pre_score,
                    next_score=next_score, review_score=review_score)

    @classmethod
    def get_final_train_set(cls):
        train_num = int(0.7*cls.LABELED_NUM)
        print train_num
        return cls.get_joint_feature_set(0, train_num)

    @classmethod
    def get_test_set(cls):
        test_num = int(0.3*cls.LABELED_NUM)
        test_start = cls.LABELED_NUM - test_num
        print test_start
        return cls.get_joint_feature_set(test_start, cls.LABELED_NUM)

    @classmethod
    def get_joint_feature_set(cls, start, end):
        # get slice
        label_sentiment = cls.get_label_sentiment()[start:end]
        joint_set = []
        for index, row in label_sentiment.iterrows():
            featureset = cls.senti_feature_extract(row.review_id, row.sentence_id)
            label = cls.get_label(row.review_id, row.sentence_id)
            # tuple featureset and label
            joint_set.append((featureset, label))
        return joint_set

    @classmethod
    def train(cls):
        train_set = cls.get_final_train_set()
        classifier = maxent.train(train_set, cls.MAXENT_ALGORITHM, trace=0, max_iter=1000)
        # save classifier
        f = open(cls.CLASSIFIER_FILE, 'wb')
        pickle.dump(classifier, f)
        f.close()

    @classmethod
    def test(cls):
        test_set = cls.get_test_set()
        # load classifier
        f = open(cls.CLASSIFIER_FILE, 'rb')
        classifier = pickle.load(f)
        f.close()
        print test_set
        count = 0.0
        right = 0.0
        # classify
        for featureset, label in test_set:
            label_c = classifier.classify(featureset)
            if label == label_c:
                right += 1
            count += 1
        print right, count, round(right/count, 3)


class SentimentClassifyChange(SentimentClassify):

    @classmethod
    def senti_feature_extract(cls, review_id, sentence_id):
        fset = SentimentClassify.senti_feature_extract(review_id, sentence_id)
        return dict(overall_stars=fset['overall_stars'],
                    this_r_score=fset['this_score'][0], this_p_score=fset['this_score'][1],
                    pre_r_score=fset['pre_score'][0], pre_p_score=fset['pre_score'][1],
                    next_r_score=fset['next_score'][0], next_p_score=fset['next_score'][1],
                    review_r_score=fset['review_score'][0], review_p_score=fset['review_score'][1])


def main():

    print


if __name__ == "__main__":
    main()
