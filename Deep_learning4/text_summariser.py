import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import re
nltk.download('stopwords')
nltk.download('punkt_tab')


from google.colab import files

uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
df.head()


import random

i=random.randint(0,len(df)-1)
df['article_text'][i]


from nltk.tokenize import sent_tokenize

sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]


clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ", regex=True)
clean_sentences = [s.lower() for s in clean_sentences]
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(clean_sentences)


from sklearn.metrics.pairwise import cosine_similarity

sim_mat = cosine_similarity(sentence_vectors)


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


i = random.randint(0, len(df)-1)

print("ARTICLE:".center(50))
print('\n')
print(df['article_text'][i])
print('\n')
print("SUMMARY:".center(50))
print('\n')
print(ranked_sentences[i][1])