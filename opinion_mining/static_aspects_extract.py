from opinion_mining.text_extract import TextExtract as te
from opinion_mining.sentence_parse import ParseSentence as ps
from collections import defaultdict
from nltk.classify import MaxentClassifier as maxent
import pickle


class StaticAspectsExtract:
    """
    aspect label: 1 food, 2 service, 3 value, 4 decor, 5 other
    classifier is binary, just determine be or not be certain aspect, so we need to train 4 classifiers
    """
    LABELED_NUM = 600
    CLASSIFIER_PATH = '../saved_classifiers/'
    CLASSIFIER_NAME = '_aspect_classifier.pickle'

    def __init__(self):
        pass

    @classmethod
    def get_features(cls, aspect):
        """
        :param aspect:int. aspect label: 1 food, 2 service, 3 value, 4 decor, 5 other
        :return:
        """
        labeled_sent = te.read_sentences_from_aspects_label()
        feature_list = []
        for index, row in labeled_sent.iterrows():
            word_dict = defaultdict(int)
            label = int(str(aspect) in row['aspects'])
            for word in ps.word_tokenize(row['sentence']):
                if ps.check_stopwords(word) or word in ['.', '?', '!', ',', '\'\'', '``']:
                    continue
                word_dict[ps.snowball_stem(word)] += 1
            feature_list.append((dict(word_dict), label))
        return feature_list

    @classmethod
    def get_classifier_name(cls, aspect):
        aspect_dict = {'1': 'food', '2': 'service', '3': 'value', '4': 'decor', '5': 'other'}
        name_list = [cls.CLASSIFIER_PATH, aspect_dict[str(aspect)], cls.CLASSIFIER_NAME]
        return ''.join(name_list)

    @classmethod
    def train(cls, aspect):
        print cls.get_features(aspect)
        print cls.get_classifier_name(aspect)
        train_set = cls.get_features(aspect)[:int(0.7*cls.LABELED_NUM)]
        classifier = maxent.train(train_set, 'IIS', trace=0, max_iter=1000)
        # save classifier
        f = open(cls.get_classifier_name(aspect), 'wb')
        pickle.dump(classifier, f)
        f.close()

    @classmethod
    def test(cls, aspect):
        test_num = int(0.3*cls.LABELED_NUM)
        test_start = cls.LABELED_NUM - test_num
        test_set = cls.get_features(aspect)[test_start: cls.LABELED_NUM]
        # load classifier
        f = open(cls.get_classifier_name(aspect), 'rb')
        classifier = pickle.load(f)
        f.close()
        count = 0.0
        right = 0.0
        # classify
        for featureset, label in test_set:
            label_c = classifier.classify(featureset)
            if label == label_c:
                right += 1
            count += 1
        print right, count, round(right/count, 3)





