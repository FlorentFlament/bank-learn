#!/usr/bin/python3
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_lines(filename):
    lines = None
    with open(filename) as fd:
        fd.readline() # Consume header line
        lines = [s.strip() for s in fd.readlines()]
    return lines

def random_training(filename, outname, count=10):
    lines = read_lines(filename)
    random.shuffle(lines)
    with open(outname, "a") as fd:
        for l in lines[:count]:
            print(l.replace(';', '\n'))
            category = input("> ")
            fd.write("{};{}\n".format(l.strip(),category))    

def read_training_set(filename):
    xy = read_lines(filename)
    x,y = [],[]
    for l in xy:
        toks = l.split(';')
        x.append(";".join(toks[:-1]))
        y.append(toks[-1])
    return x, y



training_fname = sys.argv[1]
prod_fname = sys.argv[2]

#random_training(prod_fname, training_fname)
#exit()

train_x, train_y = read_training_set(training_fname)

# Bag of Words representation
# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(train_x)

# Using Naive Bayes classifier for multinomial models
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
clf = MultinomialNB()
clf.fit(train_X, train_y)

prod_x = read_lines(prod_fname)
prod_X = vectorizer.transform(prod_x)
pred = clf.predict(prod_X)

print("date;type0;type1;details;amount;category")
for x,y in zip(prod_x, pred):
    print("{};{}".format(x,y))
