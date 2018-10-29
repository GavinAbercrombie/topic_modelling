# -*- coding: utf-8 -*-

import csv, sys, re
import numpy as np
import textacy
import spacy
import pandas as pd

csv.field_size_limit(sys.maxsize)

nlp = spacy.load('en')

debates = []
data = csv.reader(open('../debate_csvs/HanDeSeT.csv', 'r'))
for row in data:
	# adapted for handeset which features 7 columns of text per document (row)
	debates.append([row[1], row[6] + row[7] + row[8] + row[9] +row[10] +row[11]])

df = pd.DataFrame(debates, columns=['title', 'text'])
chat_concat = (df.sort_values('title').groupby('title')['text'].agg(lambda col: '\n'.join(col.astype(str))))
docs = list(chat_concat.apply(lambda x: nlp(x)))
corpus = textacy.corpus.Corpus(nlp, docs=docs)
vectorizer = textacy.Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth', norm='l2', min_df=2, max_df=5)
doc_term_matrix = vectorizer.fit_transform((doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) 
                                            for doc in corpus))
model = textacy.TopicModel('nmf', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)

for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=10):
	print('topic', topic_idx, ':', '   '.join(top_terms))
