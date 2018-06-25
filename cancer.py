# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:19:17 2018

@author: AlexWang
"""
#%%
from tkinter import *

import csv
import nltk
import re

stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

doc_list = []
with open('text.csv', 'r',encoding = 'latin-1') as f:
    reader = csv.reader(f)
    for i in reader:
        doc_list.append(i[0])

print(tokenize_and_stem(doc_list[0]))

from gensim.models import doc2vec
from collections import namedtuple

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc_list):
    words = tokenize_and_stem(text)
    tags = [i]
    docs.append(analyzedDocument(words, tags))


from gensim.models import doc2vec
import random

alpha_val = 0.025        # Initial learning rate
min_alpha_val = 1e-4     # Minimum for linear learning rate decay

model = doc2vec.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)

model.build_vocab(docs) # Building vocabulary


model.train(docs, total_examples=model.corpus_count, epochs=model.epochs, start_alpha = alpha_val, end_alpha = min_alpha_val)
print(model.epochs)
#%%
def search():
    text = getQuery()
    doc_num = getNumber()
    tokens = tokenize_and_stem(text)
    new_vector = model.infer_vector(tokens)
    sims = model.docvecs.most_similar([new_vector], topn = doc_num)
    sent = "Top " + str(doc_num) + "most relevant blogs are: " + '\n' +'\n'

    for i in sims:
        sent += doc_list[i[0]]
        sent += '\n'
        sent += '\n'
    return sent    
    
def getNumber():
    num = docnum.get()
    num = int(num)
    return num

def getQuery():
    text = query.get()
    return text

def display():
    txt.insert(0.0, search())
    
def clear():
    txt.delete('1.0',END)

root = Tk()
frame = Frame(root, height = 1000, width = 1000)
frame.pack(side = BOTTOM)

root.title('Cancer Blog Decision Support')

title = Label(root, text = "Cancer Blog Decision Support", bg ="pink", fg = "white", font = (None, 20))
title.pack(fill = X)


query = StringVar()
docnum = StringVar()

entry = Entry(frame, textvariable = query)
query_label = Label(frame, text = "Enter your query:")
entry2 = Entry(frame, textvariable = docnum)
docnum_label = Label(frame, text = "Choose Number of Docs:")
search_button = Button(frame, text = "Search", command = display)
clear_button = Button(frame, text = "Clear", command = clear)



query_label.grid(row = 0)
entry.grid(row = 0, column = 1)
docnum_label.grid(row = 1)
entry2.grid(row = 1, column = 1)
search_button.grid(row = 1, column = 2)
clear_button.grid(row = 1, column = 3)

txt = Text(frame, width = 80, height = 50, wrap = WORD)
txt.grid(row = 3, columnspan = 2, sticky = W)

root.mainloop()


