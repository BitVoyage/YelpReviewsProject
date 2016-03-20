import sys
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import os
from nltk.parse import stanford


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
    def snowball_stem(cls, word):
        snowball = SnowballStemmer('english')
        return snowball.stem(word)

    @classmethod
    def check_stopwords(cls, word):
        if not len(cls.STOPWORDS):
            cls.STOPWORDS = set(stopwords.words('english'))
        return word in cls.STOPWORDS

    @classmethod
    def word_tokenize(cls, sent):
        return word_tokenize(sent)

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

