import gensim
from gensim.models.doc2vec import TaggedDocument

from gensim.models import KeyedVectors
w2v_g100b = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_g100b.compact_name = 'w2v_g100b'
word_models.append(w2v_g100b)

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text



model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
model.save("doc2vec")