#!/usr/bin/python3
import random
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def read_lines(filename):
    lines = None
    with open(filename) as fd:
        lines = [s.strip() for s in fd.readlines()]
    return lines


class Corpus:
    def __predict(self):
        ## TODO That doesn't need to be performed each time we call __learn
        x,y = [],[]
        for l in self.__training_set:
            toks = l.split(';')
            x.append(";".join(toks[:-1]))
            y.append(toks[-1])

        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        self.__text_clf.fit(x, y)
        self.__prediction = self.__text_clf.predict(self.__corpus)
        print(self.__prediction)

    def __init__(self, training_fname, corpus_fname):
        self.__training_set = read_lines(training_fname)
        self.__corpus = read_lines(corpus_fname)
        self.__text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])
        self.__predict()

    # Commands
    ##########

    def c_save_prediction(self, filename):
        with open(filename, "w") as fd:
            for x,y in zip(self.__corpus, self.__prediction):
                fd.write("{};{}\n".format(x,y))

    def c_save_training(self, filename):
        with open(filename, "w") as fd:
            for ln in self.__training_set:
                fd.write("{}\n".format(ln))

    def c_overview(self):
        vals = [float(s.split(';')[-1].replace(' ','')) for s in self.__corpus]
        percat = {}
        for x,y in zip(vals, self.__prediction):
            percat[y] = percat.get(y, 0) + x
        for c,v in sorted(percat.items(), key=lambda x:x[1]):
            print("{:<10} {}".format(round(v,2), c))

    def c_list_category(self, category):
        percat = {}
        for x,y in zip(self.__corpus, self.__prediction):
            c = percat.setdefault(y, [])
            c.append(x)
        for item in percat[category]:
            # TODO Use dictionary to fetch item index
            print("{:<5} {}".format(self.__corpus.index(item), item))

    def c_categorize(self, item_id, category):
        item = self.__corpus[item_id]
        self.__training_set.append(';'.join([item, category]))
        self.__predict()


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
