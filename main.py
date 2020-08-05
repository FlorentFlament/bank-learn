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
    def c_overview(self):
        """Overview command"""
        pred = self.__model.predict(self.__corpus_vec)
        print("date;type0;type1;details;amount;category")
        for x,y in zip(self.__corpus_str, pred):
            print("{};{}".format(x,y))


training_fname = sys.argv[1]
corpus_fname   = sys.argv[2]
corp = Corpus(training_fname, corpus_fname)
corp.c_overview()
