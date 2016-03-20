# encoding='utf-8'
import pandas as pd
import random
import nltk.tokenize


class TextExtract:
    """
    Extract raw data to required content
    """
    PATH = '/home/zls/Projects/yelp_dataset/'
    review_filename = 'high_review_restaurants1.csv'
    train_set_filename = 'sentiment_train_set.csv'
    label_sentiment_filename = 'label_sentiment.txt'
    label_aspects_filename = 'label_aspects.txt'
    # for removing sentences shorter than threshold
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
        sentences = u''.join(sentence_array).decode('utf-8').replace('\n', '').lower()
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

    @classmethod
    def read_sentences_from_aspects_label(cls):
        sentence_list = []
        with open(cls.PATH + cls.label_aspects_filename, 'r') as rows:
            for row in rows:
                r = row.strip().split('||')
                row_dict = {'aspects': r[0].split(','), 'sentence': r[1]}
                sentence_list.append(row_dict)
        sentence_df = pd.DataFrame(sentence_list)
        return sentence_df


