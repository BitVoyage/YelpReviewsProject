# encoding='utf-8'
import pandas as pd
import random
import nltk.tokenize
import sys
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import os
from nltk.parse import stanford


class TextExtract:
    """
    Extract raw data to required content
    """
    PATH = '/home/zls/Projects/yelp_dataset/'
    review_filename = 'high_review_restaurants1.csv'
    train_set_filename = 'sentiment_train_set.csv'
    label_sentiment_filename = 'label_sentiment.txt'
    label_aspects_filename = 'label_aspects.txt'
    # for remove sentences shorter than threshold
    SENT_LENGTH_THRESHOLD = 5
    sent_detector = None

    def __init__(self):
        pass

    @classmethod
    def read_reviews(cls, random_rows=None):
        """
        get all / random selection reviews from file
        :param random_rows: int, default None. Number of random rows of file to read
        :return: dataframe. reviews from file
        """
        f_reviews = pd.read_csv(cls.PATH+cls.review_filename, encoding='utf-8')
        if random_rows:
            random_index = random.sample(f_reviews.index, random_rows)
            return f_reviews.ix[random_index]

        return f_reviews

    @classmethod
    def get_sent_detector(cls):
        if cls.sent_detector is None:
            cls.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return cls.sent_detector

    @classmethod
    def get_sentences(cls, reviews):
        """
        get tokenized sentences from reviews array
        :param reviews: Series. reviews for extract sentences
        :return: list. Sentences extracted
        """
        sent_detector = cls.get_sent_detector()

        sentence_array = reviews.text
        # prepare for tokenize, join array to string
        sentences = u''.join(sentence_array).decode('utf-8').replace('\n', '')
        sent_list = sent_detector.tokenize(sentences)
        # remove sentence whose length is shorter than threshold
        sent_list_filtered = [sent for sent in sent_list if len(sent) > cls.SENT_LENGTH_THRESHOLD]
        return sent_list_filtered

    @classmethod
    def create_sentiment_sentences_dataframe(cls, reviews):
        """
        original reviews headers:
        -- Unnamed: 0,business_id,date,review_id,stars,text,type,user_id,votes,name,
        -- attributes,categories,review_count,overall_stars
        target reviews with sentiment headers:
        -- business_id, review_id, overall_stars, sentence_id, sentence, pos_neu_neg,
        -- static_aspects, dynamic_aspects, raw_score, purity_score
        :param reviews: Dataframe. mount of  reviews
        :return: Dataframe. new construct of reviews with each row presenting one single sentence
                            and this sentence's sentiment properties
        """
        # columns = ['business_id', 'review_id', 'overall_stars', 'sentence_id', 'sentence', 'pos_neu_neg',
        #           'static_aspects', 'dynamic_aspects', 'raw_score', 'purity_score']
        sentiment_sentences_list = []
        for index, review in reviews.iterrows():
            sentences = cls.get_sentences(review)
            for idx, sent in enumerate(sentences):
                # idx as sentence_id in one review, marked unique
                row = {'business_id': review.business_id, 'review_id': review.review_id,
                       'overall_stars': review.overall_stars, 'sentence_id': idx,
                       'sentence': sent, 'pos_neu_neg': None, 'static_aspects': None, 'dynamic_aspects': None,
                       'raw_score': None, 'purity_score': None}
                sentiment_sentences_list.append(row)
        # transform dictionary to DataFrame
        sentiment_sentences = pd.DataFrame(sentiment_sentences_list)
        return sentiment_sentences

    @classmethod
    def save_sentences_for_aspects_label(cls, number):
        """
        select random number of sentences to save
        :param number: int. number of sentences to be selected
        :return: Void.
        """
        f_reviews = cls.read_reviews(number)
        f_write = open(cls.PATH+cls.label_aspects_filename, 'w')
        for index, review in f_reviews.iterrows():
            # get one single sentence randomly
            print cls.get_sentences(review)
            sentence = random.choice(cls.get_sentences(review))
            # split by '||'
            f_write.write('||'+sentence+'\n')
        f_write.close()

    @classmethod
    def save_sentiment_sentences_csv(cls, reviews):
        df = cls.create_sentiment_sentences_dataframe(reviews)
        df.to_csv(cls.PATH + cls.train_set_filename)

    @classmethod
    def save_sentences_for_sentiment_label(cls):
        """
        export sentences file for labeling.
        NOTE: label 1 for positive, 2 for neutral, 3 for negative
        :return: void.
        """
        df = pd.DataFrame.from_csv(cls.PATH+cls.train_set_filename)
        # remove duplicate review ids
        review_ids = set(df.review_id)
        # prepare for write to file
        f_label = open(cls.PATH+cls.label_sentiment_filename, 'w')
        for review_id in review_ids:
            # select by same review_id
            df_same_review = df[df['review_id'] == review_id].copy()
            # random select a row
            row_id = random.choice(df_same_review.index.values)
            df_row = df_same_review.ix[row_id].copy()
            # split by '||' ==> [sentiment, sentence, space, sentence_id, review_id]
            f_label.write('||'+df_row.sentence+'||'+' '*200+'||'+str(df_row.sentence_id)+'||'+df_row.review_id+'\n')
        f_label.close()

    @classmethod
    def read_sentences_from_trainset(cls):
        return pd.read_csv(cls.PATH + cls.train_set_filename)

    @classmethod
    def read_sentences_from_sentiment_label(cls):
        """
        read from txt and return format data
        :return: dataframe.
        """
        polar_dict = {'1': 1, '2': 0, '3': -1, '': -2}
        sentence_list = []
        with open(cls.PATH + cls.label_sentiment_filename, 'r') as rows:
            for row in rows:
                r = row.strip().split('||')
                row_dict = {'pos_neu_neg': polar_dict[r[0]], 'sentence': r[1], 'sentence_id': r[3],
                            'review_id': r[4]}
                sentence_list.append(row_dict)
        sentence_df = pd.DataFrame(sentence_list)
        return sentence_df


class ParseSentence:

    # pos map from simple to sentiWordNet type
    pos_dict = {'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r', 'VERB': 'v'}
    STOPWORDS = set()

    PATH_TO_STANFORD = '../extern/stanford-parser-full-2015-12-09/'
    os.environ['STANFORD_PARSER'] = PATH_TO_STANFORD + 'stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = PATH_TO_STANFORD + 'stanford-parser-3.6.0-models.jar'

    NEGATION_SET = set([u"not", u"n't"])
    NEGATION_DISTANCE = 4

    _dep_parser = None

    def __init__(self):
        pass

    @classmethod
    def get_dep_parser(cls):
        if cls._dep_parser is None:
            cls._dep_parser = stanford.StanfordDependencyParser(model_path=cls.PATH_TO_STANFORD + "englishPCFG.ser.gz")
        return cls._dep_parser

    @classmethod
    def parse_sent_for_sentiwordnet(cls, sentence):
        """
        tokenize ==> pos_tag ==> reduce to nouns(n), adjectives(a), verbs(v), adverbs(r) ==> normalization:stemming
        :param sentence:
        :return:
        """
        # Note: lower sentence
        sentence = str.lower(sentence)

        word_tokenized_sent = word_tokenize(sentence)
        pos_words = pos_tag(word_tokenized_sent, tagset='universal')
        # Tuple can't be change values. transform tuple to list.
        tag_words_list = [[word[0], word[1], idx] for idx, word in enumerate(pos_words)]
        tag_words_list = cls.simple_negation_mark(word_tokenized_sent, tag_words_list)

        if not len(cls.STOPWORDS):
            cls.STOPWORDS = set(stopwords.words('english'))
        # remove stop words and keep only sentiment words
        tag_words_list = filter(lambda x: x[0] not in cls.STOPWORDS and x[1] in cls.pos_dict, tag_words_list)

        # stem word and change pos_type to be compatibility with sentiwordnet
        tag_words_list = [(cls.stem_word(word[0]), cls.pos_dict[word[1]], word[2]) for word in tag_words_list]

        return tag_words_list

    @classmethod
    def simple_negation_mark(cls, word_tokenized_sent, tag_word_list):
        negation_list = cls.simple_negation_detect(word_tokenized_sent)
        if not len(negation_list):
            return tag_word_list

        neg_mark_index = set()
        for neg_idx in negation_list:
            neg_mark_index = neg_mark_index | set(cls.get_neg_mark_index(neg_idx, len(tag_word_list) - 1))

        for idx in neg_mark_index:
            tag_word_list[idx][2] = -1
        return tag_word_list

    @classmethod
    def get_neg_mark_index(cls, neg_idx, max_index):
        range_raw_idx = range(neg_idx - cls.NEGATION_DISTANCE + 1, neg_idx + cls.NEGATION_DISTANCE)
        return filter(lambda x: 0 <= x <= max_index, range_raw_idx)

    @classmethod
    def simple_negation_detect(cls, word_tokenized_sent):
        negation_list = []
        for idx, word in enumerate(word_tokenized_sent):
            if word in cls.NEGATION_SET:
                negation_list.append(idx)
        return negation_list

    @classmethod
    def stanford_dependency_parse(cls, sent):
        """
        sentence tree and dependency word pair as ('bad', 'neg', 'not')
        :param sent: string. raw sentence string
        :return: list of tuples.
        """
        # TODO speed up using map or hadoop
        dep_parser = cls.get_dep_parser()
        return [list(parse.triples()) for parse in dep_parser.raw_parse(sent)][0]

    @classmethod
    def get_neg_words_list(cls, sent_list):
        """
        sentence tree and dependency word pair as ('bad', 'neg', 'not')
        :param sent_list: string. raw sentence string
        :return: list of tuples.
        """
        return map(cls.stanford_dependency_parse, sent_list)

    @classmethod
    def get_neg_words(cls, sent):
        """
        get negation words, such as word 'good' in sentence 'it's not good'
        :param sent: string. raw sentence string
        :return: list. neg words detected
        """
        parsed_sent = cls.stanford_dependency_parse(sent)
        # dependency pair as ('word modified', 'modify type', 'modifier word')
        return [neg_word[0][0] for neg_word in parsed_sent if neg_word[1] == 'neg']

    @classmethod
    def stem_word(cls, word):
        """
        stem word and keep wordnet compatibility
        :param word:string. primary word
        :return:string. stemmed word
        """
        # snowball = SnowballStemmer('english')
        return WordNetLemmatizer().lemmatize(word)

    @classmethod
    def parse_sentence(cls, sent):
        if not len(cls.STOPWORDS):
            cls.STOPWORDS = set(stopwords.words('english'))
        pos_words = pos_tag(word_tokenize(sent), tagset='universal')
        tag_words_list = [[word[0], word[1]] for word in pos_words]
        for word in tag_words_list:
            if word[0] in cls.STOPWORDS:
                tag_words_list.remove(word)
                continue
            word[0] = cls.stem_word(word[0])
        return tag_words_list

def main():
        # TODO encoding utf-8 in a more elegant way
        reload(sys)
        sys.setdefaultencoding('utf-8')
        # te = TextExtract()
        # ----don't do again, rewrite---------- prepare for aspects label -------------------
        # te.save_sentences_for_aspects_label(1000)
        # ----don't do again, rewrite ---------- prepare for sentiment label -----------------
        # reviews = te.read_reviews(1000)
        # te.save_sentiment_sentences_csv(reviews)
        # te.save_sentences_for_sentiment_label()
        # ------------------------------------------------------------

if __name__ == "__main__":
    main()

