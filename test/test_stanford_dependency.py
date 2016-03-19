import os
from nltk.parse import stanford
PATH = '../extern/stanford-parser-full-2015-12-09/'
os.environ['STANFORD_PARSER'] =PATH + 'stanford-parser.jar'
os.environ['STANFORD_MODELS'] =PATH + 'stanford-parser-3.6.0-models.jar'

# parser = stanford.StanfordParser(model_path=PATH + "englishPCFG.ser.gz")
# sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your nosetraceme?"))
# print sentences
#
# for line in sentences:
#     for sentence in line:
#        print sentence

# dependency = stanford.StanfordDependencyParser(model_path=PATH + "englishPCFG.ser.gz")
dep_parser=stanford.StanfordDependencyParser(model_path=PATH + "englishPCFG.ser.gz")
sent1 = "The quick brown foxes don't like the lazy dog which are not tall."
sent = "it's not so bad"
parsed_sent = [list(parse.triples()) for parse in dep_parser.raw_parse(sent)][0]
print parsed_sent
print [neg_word[0][0] for neg_word in parsed_sent if neg_word[1] == 'neg']
