import nltk
from collections import Counter
from opinion_mining.text_extract import TextExtract
from opinion_mining.sentence_parse import ParseSentence


class DynamicAspectsExtract:

    _np_pattern = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>}
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                """
    _nouns_pattern = r"""
                    CP:  {<NN.*>*}
                    """

    _MAX_NOUNS = 3
    _ASPECT_MIN_FREQ = 5
    _except_words = ['food', 'value', 'service', 'server', 'place', 'visit',
                     'lot', 'thing', 'restaurant', 'anything', 'something', 'time',
                     'everyone', 'everything', 'review', 'people', 'line']
    _business_id = 0

    def __init__(self, business_id):
        self._business_id = business_id

    def set_np_pattern(self, np_pattern):
            self._np_pattern = np_pattern

    def __syntactic_parse(self, sent):
        chunker = nltk.RegexpParser(self._np_pattern)
        sent_pos = nltk.pos_tag(nltk.word_tokenize(sent))
        tree = chunker.parse(sent_pos)
        return tree

    @classmethod
    def __get_chunks(cls, sent_tree, chunk_type):
        chunks = []
        for subtree in sent_tree.subtrees(filter=lambda t: t.label() == chunk_type):
            chunks.append(subtree.leaves())
        return chunks

    @classmethod
    def __get_dry_phase(cls, nouns_list):
        nouns = [ParseSentence.stem_word(nouns[0]) for nouns in nouns_list]
        phase = ' '.join(nouns).strip()
        return phase

    def get_candidate_aspects(self, sent):
        nouns_chunker = nltk.RegexpParser(self._nouns_pattern)
        sent_tree = self.__syntactic_parse(sent)
        candi_aspects = []
        for np in self.__get_chunks(sent_tree, 'NP'):
            np_tree = nouns_chunker.parse(np)
            nouns_list = self.__get_chunks(np_tree, 'CP')
            if not len(nouns_list):
                continue
            if len(nouns_list) <= self._MAX_NOUNS:
                candi_aspects.append(self.__get_dry_phase(nouns_list[0]))
        return candi_aspects

    def is_illegal(self, word):
        return ParseSentence.check_stopwords(word) or word in self._except_words

    def get_aspects(self):
        df = TextExtract.read_sentences_from_trainset()
        sentences = df[df['business_id'] == self._business_id].sentence
        raw_aspects = []
        aspect_counter = Counter()
        for sent in sentences:
            candi_aspect = self.get_candidate_aspects(sent.lower())
            raw_aspects.extend(candi_aspect)
        for aspect in raw_aspects:
            if not self.is_illegal(aspect):
                aspect_counter[aspect] += 1
        print [(aspect, freq) for aspect, freq in aspect_counter.most_common(3) if freq > self._ASPECT_MIN_FREQ]



