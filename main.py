#!/usr/bin/python3
import random
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_lines(filename):
    lines = None
    with open(filename) as fd:
        fd.readline() # Consume header line
        lines = [s.strip() for s in fd.readlines()]
    return lines


class Corpus:
    def __init_model(self, filename):
        xy = read_lines(filename)
        x,y = [],[]
        for l in xy:
            toks = l.split(';')
            x.append(";".join(toks[:-1]))
            y.append(toks[-1])

        # Bag of Words representation
        # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
        X = self.__vectorizer.fit_transform(x)

        # Using Naive Bayes classifier for multinomial models
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
        self.__model = MultinomialNB()
        self.__model.fit(X, y)

    def __init_corpus(self, filename):
        self.__corpus_str = read_lines(filename)
        self.__corpus_vec = self.__vectorizer.transform(self.__corpus_str)

    def __init__(self, training_fname, corpus_fname):
        self.__vectorizer = CountVectorizer()
        self.__init_model(training_fname)
        self.__init_corpus(corpus_fname)

    # Commands
    def c_save_prediction(self, filename):
        """Overview command"""
        pred = self.__model.predict(self.__corpus_vec)
        with open(filename, "w") as fd:
            for x,y in zip(self.__corpus_str, pred):
                fd.write("{};{}\n".format(x,y))

    def c_overview(self):
        pred = self.__model.predict(self.__corpus_vec)
        vals = [float(s.split(';')[-1].replace(' ','')) for s in self.__corpus_str]
        percat = {}
        for x,y in zip(vals, pred):
            percat[y] = percat.get(y, 0) + x
        for c,v in sorted(percat.items(), key=lambda x:x[1]):
            print("{:<10} {}".format(round(v,2), c))

training_fname = sys.argv[1]
corpus_fname   = sys.argv[2]
prediction_fname = sys.argv[3]

corp = Corpus(training_fname, corpus_fname)
cmd = None
while cmd != 'q':
    ln = input("> ").split()
    cmd = ln[0]
    if cmd == 'o':
        corp.c_overview()
