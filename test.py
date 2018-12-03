
import time, re
import math

import nltk

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def split_article(article):
  words=["Copyright ","©","elsevier inc rights reserved", "elsevier science inc", "elsevier science ltd"]
  result=article
  for word in words:
    result=result.split(word)[0]
  return result

def article_count(pattern, articles):
  return len([1 for a in articles if pattern in a.lower()])

def getWords(s):
  exclusion = re.findall("([A-Za-z]+\d+)\s+",s)
  words = re.compile("[\-\d:\\\"\s\.\,\(\)\%';\[\]]").split(s.lower())
  # print(len(words))
  return list(filter(lambda a: a not in stopwords, words))


def load_tfidfmodel(articles, data):

  from sklearn.feature_extraction.text import TfidfVectorizer
  from collections import defaultdict

  mindf = int(len(articles)*0.00125)

  vectorizer = TfidfVectorizer(ngram_range=(3,5), max_df=0.05, min_df=mindf)

  X = vectorizer.fit_transform(articles)

  features_by_gram = defaultdict(list)

  for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram[len(f.split(' '))].append((f, w))

  # print(features_by_gram)

  top_n = 100
  _features = []
  for gram, features in features_by_gram.items():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [(f[0], article_count(f[0], articles)) for f in top_features]
    _features += sorted(top_features, key=lambda x: x[1], reverse=True)

  _filtred_features = []
  for f, i in _features:
    skip=False
    for l, j in _features:
      if i!=j:
        if l in f:
          skip=True

    if not skip:
      _filtred_features.append((f, i))

  sorted_features = sorted(_filtred_features, key=lambda x: x[1], reverse=True)

  for f in sorted(_filtred_features, key=lambda x: x[1], reverse=True):
    print("{:<50} : {:<4}".format(f[0], f[1]))


stopwords = nltk.corpus.stopwords.words('english')
stopwords += ['elsevier', 'rights', 'reserved', 'inc', 'ltd']
data = pd.read_csv("data/scientometrics_literature_1945_2014.csv", delimiter="\t",error_bad_lines=False).fillna('')

data['AB'] = data['AB'].apply(split_article)


b = [" ".join(getWords(a)) for a in data['AB']]

load_tfidfmodel(b, data)

# prepare_report('', data)


