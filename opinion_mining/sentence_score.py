# coding=utf-8
from nltk.corpus import sentiwordnet as swn
from sentence_parse import ParseSentence as ps


class ScoreBasedOnSentiWordNet:

    PATH_TO_LEXICONS = "../data"
    _special_words = None

    def __init__(self):
        pass

    @classmethod
    def parse_special_sentiword(cls, row):
        word, pos_score, neg_score = row.strip().split()
        return word, float(pos_score) - float(neg_score)

    @classmethod
    def read_special_word(cls):
        if cls._special_words is None:
            with open(cls.PATH_TO_LEXICONS+'/special_sentiword_for_restaurant.txt', 'r') as rows:
                cls._special_words = dict(cls.parse_special_sentiword(row) for row in rows)
        return cls._special_words

    @classmethod
    def get_score(cls, sentence):
        """
        score pair tuple: (raw_score, purity_score)
        raw score: sum of every word's scores with pos/neg sign, word's score is reversed when negation is detected
        purity score: raw score / (simply sum of word's ABS scores)
        :param sentence: string. raw sentence string.
        :return: int. score of sentence
        """
        parsed_sent = ps.parse_sent_for_sentiwordnet(sentence)
        neg_words = ps.get_neg_words(sentence)
        stemmed_neg_words = [ps.stem_word(word) for word in neg_words]
        raw_score = 0
        abs_score = 0
        for word in parsed_sent:
            primary_score = cls.get_word_primary_score(word)
            abs_score += abs(primary_score)
            # reverse score if word is negation
            if word[0] in stemmed_neg_words:
                raw_score -= primary_score
            else:
                raw_score += primary_score
        # avoid 0 division
        if abs_score == 0:
            return 0, 0
        return raw_score, round(raw_score/abs_score, 3)

    @classmethod
    def get_score_simple_neg(cls, sentence):
        """
        score pair tuple: (raw_score, purity_score)
        raw score: sum of every word's scores with pos/neg sign, word's score is reversed when negation is detected
        purity score: raw score / (simply sum of word's ABS scores)
        :param sentence: string. raw sentence string.
        :return: int. score of sentence
        """
        parsed_sent = ps.parse_sent_for_sentiwordnet(sentence)
        raw_score = 0
        abs_score = 0
        for word in parsed_sent:
            primary_score = cls.get_word_primary_score(word)
            abs_score += abs(primary_score)
            # reverse score if word is negation
            if word[2] == -1:
                raw_score -= primary_score
            else:
                raw_score += primary_score
        # avoid 0 division
        if abs_score == 0:
            return 0, 0
        return round(raw_score, 3), round(raw_score/abs_score, 3)

    @classmethod
    def get_word_primary_score(cls, word_pair):
        # firstly checking the special words
        special_words = cls.read_special_word()
        if word_pair[0] in special_words:
            return special_words[word_pair[0]]
        # not in sentiwordnet then return 0
        synsets = swn.senti_synsets(word_pair[0], word_pair[1])
        if not len(synsets):
            return 0

        score_list = [synset.pos_score() - synset.neg_score() for synset in synsets]
        return sum(score_list) / float(len(score_list))


class ScoreBasedOnAFINN:
    """
    Class for scoring sentences based on AFINN lexicon.

    AFINN is a list of English words rated for valence with an integer
    between minus five (negative) and plus five (positive). The words have
    been manually labeled by Finn Årup Nielsen in 2009-2011.

    """

    PATH_TO_LEXICONS = "../data/AFINN-111.txt"

    _afinn = None

    def __init__(self):
        """
        Read in the lexicons.
        """
        self.afinn = self.read_lexicon()

    @classmethod
    def afinn_parse(cls, row):
        word, score = row.strip().split('\t')
        return word, int(score)

    @classmethod
    def read_lexicon(cls):
        # Singleton Model
        if cls._afinn is None:
            path = cls.PATH_TO_LEXICONS
            with open(path, 'r') as rows:
                cls._afinn = dict(cls.afinn_parse(row) for row in rows)
        return cls._afinn

    @classmethod
    def predict(cls, tokenized_sent):
        doc_len = len(tokenized_sent)
        assert doc_len > 0, "Can't featurize document with no tokens."

        tokenized_sent = [ps.stem_word(word) for word in tokenized_sent]
        afinn = cls.read_lexicon()
        score_raw = sum([afinn[tok] if tok in afinn else 0 for tok in tokenized_sent])
        score_weight = sum([abs(afinn[tok]) if tok in afinn else 0 for tok in tokenized_sent])

        if score_weight == 0:
            score = 0
        else:
            score = score_raw / score_weight

        return score


class ScoreBasedOnBingLiu:
    """
    Class for scoring sentences using Bing Liu's Opinion Lexicon.

    Source:

    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
    Proceedings of the ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
    Washington, USA,

    Download lexicon at: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
    """

    PATH_TO_LEXICONS = "../data"

    _pos_lex = None
    _neg_lex = None

    def __init__(self):
        """
        Read in the lexicons.
        """
        self._pos_lex = self.read_pos()
        self._neg_lex = self.read_neg()

    @classmethod
    def read_pos(cls):
        if cls._pos_lex is None:
            pos_path = cls.PATH_TO_LEXICONS + "/positive-words.txt"
            cls._pos_lex = cls.read_lexicon(pos_path)
        return cls._pos_lex

    @classmethod
    def read_neg(cls):
        if cls._neg_lex is None:
            neg_path = cls.PATH_TO_LEXICONS + "/negative-words.txt"
            cls._neg_lex = cls.read_lexicon(neg_path)
        return cls._neg_lex

    @classmethod
    def read_lexicon(cls, path):
        start_read = False
        lexicon = set()  # set for quick look-up

        with open(path, 'r') as f:
            for line in f:
                if start_read:
                    lexicon.add(line.strip())
                if line.strip() == "":
                        start_read = True
        return lexicon

    @classmethod
    def predict(cls, tokenized_sent):
        # features = {}
        doc_len = len(tokenized_sent)
        assert doc_len > 0, "Can't featurize document with no tokens."

        tokenized_sent = [ps.stem_word(word) for word in tokenized_sent]
        pos_lex = cls.read_pos()
        neg_lex = cls.read_neg()
        num_pos = sum([1 if tok in pos_lex else 0 for tok in tokenized_sent])
        num_neg = sum([1 if tok in neg_lex else 0 for tok in tokenized_sent])

        # features['liu_pos'] = num_pos/doc_len
        # features['liu_neg'] = num_neg/doc_len

        score = (num_pos - num_neg) / doc_len
        return score

    # return 1 if score > 0.5 else 0
