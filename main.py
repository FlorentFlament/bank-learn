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
    def __learn(self):
        ## TODO That doesn't need to be performed each time we call __learn
        x,y = [],[]
        for l in self.__training_set:
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

        # Project self.__corpus_vec on features extracted by the vectorizer
        self.__corpus_vec = self.__vectorizer.transform(self.__corpus_str)

    def __init__(self, training_fname, corpus_fname):
        self.__vectorizer = CountVectorizer()
        self.__corpus_str = read_lines(corpus_fname)
        self.__training_set = read_lines(training_fname)
        self.__learn()

    # Commands
    ##########

    def c_save_prediction(self, filename):
        """Overview command"""
        pred = self.__model.predict(self.__corpus_vec)
        with open(filename, "w") as fd:
            for x,y in zip(self.__corpus_str, pred):
                fd.write("{};{}\n".format(x,y))

    def c_save_training(self, filename):
        """Overview command"""
        with open(filename, "w") as fd:
            for ln in self.__training_set:
                fd.write("{}\n".format(ln))

    def c_overview(self):
        pred = self.__model.predict(self.__corpus_vec)
        vals = [float(s.split(';')[-1].replace(' ','')) for s in self.__corpus_str]
        percat = {}
        for x,y in zip(vals, pred):
            percat[y] = percat.get(y, 0) + x
        for c,v in sorted(percat.items(), key=lambda x:x[1]):
            print("{:<10} {}".format(round(v,2), c))

    def c_list_category(self, category):
        pred = self.__model.predict(self.__corpus_vec)
        percat = {}
        for x,y in zip(self.__corpus_str, pred):
            c = percat.setdefault(y, [])
            c.append(x)
        for item in percat[category]:
            # TODO Use dictionary to fetch item index
            print("{:<5} {}".format(self.__corpus_str.index(item), item))

    def c_categorize(self, item_id, category):
        item = self.__corpus_str[item_id]
        self.__training_set.append(';'.join([item, category]))
        self.__learn()


def help():
    msg="""h                    Display this help message
o                    Display an overview of the expenses / incomes per category
p out_fname          Write corpus with predicted categories appended to out_fname file
t out_fname          Write training set to out_fname file
l category           List entries in category
c item_id category   Classify item with id item_id to category
q                    Quit"""
    print(msg)

training_fname = sys.argv[1]
corpus_fname   = sys.argv[2]

corp = Corpus(training_fname, corpus_fname)
cmd = 'o'
while cmd != 'q':
    if cmd == '':
        pass
    elif cmd == 'o':
        print("*** Overview")
        corp.c_overview()
    elif cmd == 'p':
        try:
            out_fname = ln[1]
            print("*** Writing predictions to {}".format(out_fname))
            corp.c_save_prediction(out_fname)
        except IndexError:
            print("*** argment out_fname missing")
            help()
    elif cmd == 'l':
        try:
            category = ln[1]
            print("*** Listing items in category {}".format(category))
            corp.c_list_category(category)
        except IndexError:
            print("*** argment category missing")
            help()
        except KeyError:
            print("*** wrong category {}".format(category))
            help()
    elif cmd == 'c':
        item = int(ln[1])
        cat  = ln[2]
        print("*** Classifying {} into {}".format(item, cat))
        corp.c_categorize(item, cat)
    elif cmd == 't':
        out_fname = ln[1]
        print("*** Saving training file to {}".format(out_fname))
        corp.c_save_training(out_fname)
    elif cmd == 'h':
        print("*** Help")
        help()
    else:
        print("*** Unknown command: {}".format(cmd))
        help()
    try:
        ln = input("> ").split()
        cmd = ln[0]
    except (EOFError, KeyboardInterrupt):
        cmd = 'q'
    except IndexError:
        cmd = ''
