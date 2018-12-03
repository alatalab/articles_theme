
import numpy as np
import pandas as pd
import nltk
import gensim
from gensim import models

def parseData(file):
  articles = pd.read_csv(file)
  articles = articles[articles['Abstract'] != '[No abstract available]']
  return articles

def main():

  stopwords = nltk.corpus.stopwords.words('english')

  data = parseData("data/world_history_key.csv")

  text = data['Abstract'].str.cat(sep=" ")

  model = models.TfidfModel(text)

  model.save('w_h_tfidf.bin')

  doc_bow = [(0, 1), (1, 1)]

  print(tfidf[doc_bow])


main()

