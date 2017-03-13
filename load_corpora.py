
import time, re

import nltk
import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import numpy as np

import collections
import operator

from vocabulary import Vocabulary as vb
import json

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


# from keras


###
# hacks
###

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# input_list = ['all', 'this', 'happened', 'more', 'or', 'less']
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

###

import nltk
stopwords = nltk.corpus.stopwords.words('english')

stopwords+=["", "©", "A", "1", "2", "3", "4", "5", "6", '10', "of", "It", "In", "This", "a", "The", "the", "was", "We", "about", "above", "according", "across", "actually", "ad", "adj", "ae", "af", "after", "afterwards", "ag", "again", "against", "ai", "al", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anywhere", "ao", "aq", "ar", "are", "aren", "aren't", "around", "arpa", "as", "at", "au", "aw", "az", "ba", "bb", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "behind", "being", "below", "beside", "besides", "between", "beyond", "bf", "bg", "bh", "bi", "billion", "bj", "bm", "bn", "bo", "both", "br", "bs", "bt", "but", "buy", "bv", "bw", "by", "bz", "ca", "can", "can't", "cannot", "caption", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "click", "cm", "cn", "co", "co.", "com", "copy", "could", "couldn", "couldn't", "cr", "cs", "cu", "cv", "cx", "cy", "cz,de", "did", "didn", "didn't", "dj", "dk", "dm", "do", "does", "doesn", "doesn't", "don", "don't", "down", "during", "dz,each", "ec", "edu", "ee", "eg", "eh", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "er", "es", "et", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except,few", "fi", "fifty", "find", "first", "five", "fj", "fk", "fm", "fo", "for", "former", "formerly", "forty", "found", "four", "fr", "free", "from", "further", "fx,ga", "gb", "gd", "ge", "get", "gf", "gg", "gh", "gi", "gl", "gm", "gmt", "gn", "go", "gov", "gp", "gq", "gr", "gs", "gt", "gu", "gw", "gy ,had", "has", "hasn", "hasn't", "have", "haven", "haven't", "he", "he'd", "he'll", "he's", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "hk", "hm", "hn", "home", "homepage", "how", "however", "hr", "ht", "htm", "html", "http", "hu", "hundred", "i'd", "i'll", "i'm", "i've", "i.e.", "id", "ie", "if", "ii", "il", "im", "in", "inc", "inc.", "indeed", "information", "instead", "int", "into", "io", "iq", "ir", "is", "isn", "isn't", "it", "it's", "its", "itself", " ,je", "jm", "jo", "join", "jp,ke", "kg", "kh", "ki", "km", "kn", "koo", "kp", "kr", "kw", "ky", "kz,la", "last", "later", "latter", "lb", "lc", "least", "less", "let", "let's", "li", "like", "likely", "lk", "ll", "lr", "ls", "lt", "ltd", "lu", "lv", "ly,ma", "made", "make", "makes", "many", "maybe", "mc", "md", "me", "meantime", "meanwhile", "mg", "mh", "microsoft", "might", "mil", "million", "miss", "mk", "ml", "mm", "mn", "mo", "more", "moreover", "most", "mostly", "mp", "mq", "mr", "mrs", "ms", "msie", "mt", "mu", "much", "must", "mv", "mw", "mx", "my", "myself", "mz,na", "namely", "nc", "ne", "neither", "net", "netscape", "never", "nevertheless", "new", "next", "nf", "ng", "ni", "nine", "ninety", "nl", "no", "nobody", "none", "nonetheless", "noone", "nor", "not", "nothing", "now", "nowhere", "np", "nr", "nu", "null", "nz,of", "off", "often", "om", "on", "once", "one", "one's", "only", "onto", "or", "org", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "overall", "own,pa", "page", "pe", "per", "perhaps", "pf", "pg", "ph", "pk", "pl", "pm", "pn", "pr", "pt", "pw", "py", " ,qa,rather", "re", "recent", "recently", "reserved", "ring", "ro", "ru", "rw,sa", "same", "sb", "sc", "sd", "se", "seem", "seemed", "seeming", "seems", "seven", "seventy", "several", "sg", "sh", "she", "she'd", "she'll", "she's", "should", "shouldn", "shouldn't", "si", "since", "site", "six", "sixty", "sj", "sk", "sl", "sm", "sn", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "sr", "st", "still", "stop", "su", "such", "sv", "sy", "sz,taking", "tc", "td", "ten", "text", "tf", "tg", "test", "th", "than", "that", "that'll", "that's", "the", "their", "them", "themselves", "then", "thence", "there", "there'll", "there's", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "thirty", "this", "those", "though", "thousand", "three", "through", "throughout", "thru", "thus", "tj", "tk", "tm", "tn", "to", " together", "too", "toward", "towards", "tp", "tr", "trillion", "tt", "tv", "tw", "twenty", "two", "tz,ua", "ug", "uk", "um", "under", "unless", "unlike", "unlikely", "until", "up", "upon", "us", "use", "used", "using", "uy", "uz,va", "vc", "ve", "very", "vg", "vi", "via", "vn", "vu,was", "wasn", "wasn't", "we", "we'd", "we'll", "we're", "we've", "web", "webpage", "website", "welcome", "well", "were", "weren", "weren't", "wf", "what", "what'll", "what's", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who'd", "who'll", "who's", "whoever", " whole", "whom", "whomever", "whose", "why", "will", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "ws", "www,ye", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "yt", "yu,za", "zm", "zr"]


def main(stopwords):

  # print("check for input...")
  stopwords += generateStopBigrams()
  stopwords += nltk.corpus.stopwords.words('english')

  data = parseData("data/world_history_key.csv")

  # model = None

  model = load_model(source=data)

  model.save('world_history_2vec.bin')

  # print(model['снег_NOUN'])

  # tesaurus = parseTesaurus()

  # print(len(stopwords))


  # print(len(stopwords))


  freq_terms = None

  # for index, article in data[:5].iterrows():

  #   print("processing... {:<14}".format(article["Source title"]))



  #   freq_terms = getFreqTerms(article(["Abstract"], freq_terms)

    # meaning = getMeaning(article["Abstract"])

    # print(article)

  # text = data['Abstract'].str.cat(sep=" ")

  # print(len(text))

  # freq_terms = getFreqTerms(text)

  # print(stopwords[:-40])

  # for w in stopwords:
  #   if w in freq_terms:
  #     del freq_terms[w]

  # sorted_fr = sorted(freq_terms.items(), key=operator.itemgetter(1))

  # for k, v in sorted_fr: print(k, v)
  # _nouns = getNouns(freq_terms.items())
  # sorted_nouns = {k:v for (k,v) in sorted_fr.items() if k in _nouns}

  # for k, v in sorted_fr[-10:]: print(k, v)



  obs = []

  for i, a in data.iterrows():
    # tags = tagArticle(a)
    # print(a)
    # print(tags[:3])

    obs.append(vectorizeAbstract(a, model))



  kmeans = KMeans().fit(np.array(obs)) #k_or_guess

  # print(kmeans.cluster_centers_)


  for c in kmeans.cluster_centers_:
    topic = model.similar_by_vector(c, topn=5)
    print("-----")
    for title in topic:
      # print(title[0])
      print(title[0][:140])

def vectorizeAbstract(a, model):

  return model.docvecs[a['Title']]

  # words = getWords(a)

  # vectors = []

  # for w in words:
  #   if w.lower() in model.vocab:
  #     # print(model[w])
  #     vectors.append(model[w.lower()])

  # return sum(vectors)/len(vectors)


def generateStopBigrams():

  stop_left = ["in", "on", "to", "from", "And", "and", "as", "of"]
  stop_right = ["a", "the", "their", "them"]

  _s = []

  for l in stop_left:
    for r in stop_right:
      _s.append(l+" "+r)

  return _s

def getFreqTerms(text):
  # pass
  # if freq_terms is None: freq_terms = []

  words = getWords(text)

  f = collectFrequency(words)

  # del f['']

  _bigrams = find_ngrams(words, 2)
  bigrams = list(map(lambda a: a[0]+' '+a[1] , _bigrams))

  _trigrams = find_ngrams(words, 3)
  trigrams = list(map(lambda a: a[0]+' '+a[1]+' '+a[2] , _trigrams))

  g = collectFrequency(bigrams)
  h = collectFrequency(trigrams)

  return merge_dicts(f, g, h)


def collectFrequency(words):
  _f = dict()

  for w in words:
    _f[w] = 0
  for w in words:
    _f[w] += 1

  return _f


def getWords(s):
  words = re.compile("[:\\\"\s\.\,\(\)\%';\[\]]").split(s)
  # print(len(words))
  return list(filter(lambda a: a not in stopwords, words))

import json

thesaurus=json.load(open("thesaurus.json"))

def tagArticle(text):

  result={}

  for key in thesaurus.keys():
   # if key in text:
    qualifiers=thesaurus[key]

    if len(qualifiers )>0:

      for qualifier in qualifiers:
        if qualifier.lower() in text.lower():
          if key not in result:
            result[key]=0
          result[key]=result[key]+1

      if key in result:
        result[key]=result[key]/len(qualifiers)


  return sorted(result.items(), key=operator.itemgetter(1))

def getPartOfSpeech(a):

  test = vb.part_of_speech(a)

  if not test:
    return ''

  return json.loads(vb.part_of_speech(a))[0]["text"]


def getNouns(s):

  return list(filter(lambda a: getPartOfSpeech(a[0]) == 'noun', s))



def parseTesaurus(file="desc2017.xml"):

  start_time = time.time()

  print("parsing descriptors...")

  import xml.etree.ElementTree
  e = xml.etree.ElementTree.parse(file).getroot()

  print("descriptors loaded in %ss" % int(time.time() - start_time))

  return e



def parseData(file):

  articles = pd.read_csv(file)

  articles = articles[articles['Abstract'] != '[No abstract available]']

  # print(articles)

  return articles


def getMeaning(text, model):

  sentences = re.compile('\.').split(text)

  for s in sentences:

    words = re.compile("\s").split(s)

    _mean = np.zeros(model.dimension)

    for w in words:

      _mean += model[w]

    _mean = _mean/len(words)




class LabeledRawSentence(object):
  def __init__(self,df):
    self.df = df

  def __iter__(self):
    for index,row in self.df.iterrows():
      words = getWords(row['Abstract']+str(row['Author Keywords'])+str(row['Index Keywords']))
      if words:
        yield gensim.models.doc2vec.TaggedDocument(words, [row['Title']])


def load_model(source=None, file=None):

  if file:
    return Doc2Vec.load(file)

  # MODEL_PATH = 'models/en_1000_no_stem/en.model'
  MODEL_PATH = 'models/GoogleNews-vectors-negative300.bin'


  start_time = time.time()

  print("loading model...")

  # documents = TaggedLineDocument(sources) # automatically add line_no to each documents and enumerate it

  model = Doc2Vec(dm=0, #DV
        size=300,
        window=8,
        min_count=1, 
        iter=20,
        workers=4)
  model.build_vocab(LabeledRawSentence(source))
  model.intersect_word2vec_format(MODEL_PATH, binary=True)  # C binary format
  model.train(LabeledRawSentence(source))

  # model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

  # print("model: %s words" % len(model.vocab))


  print("model loaded in %ss" % int(time.time() - start_time))

  return model


main(stopwords)


