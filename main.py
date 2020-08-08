#!/usr/bin/env python3
import random
import re
import sys
import traceback

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Removing words that aren't relevant for classification
# Introducing more noise
STOP_WORDS=('carte', 'cb', 'du', 'facture', 'paiement')

def read_lines(filename):
    lines = []
    with open(filename) as fd:
        l = fd.readline()
        while l != "":
            if re.match("^\d{4}/\d{2}/\d{2};", l):
                # Only consider line starting with a valid date
                # This allows skipping header, and possible comments
                lines.append(l.strip())
            l = fd.readline()
    return lines

def cleaned_training_transaction(tr):
    # Keep all transaction fields except date and price, respectively
    # at 1st and last position (not using these fields for
    # categorization).
    return ';'.join(tr.split(';')[1:-1])

class Corpus:
    def __enrich_training_set(self, transaction, category):
        # Keeping all information to dump into training file
        self.__training_set_str.append("{};{}".format(transaction,category))
        # Setting x and y training set vectors
        self.__training_set_x.append(cleaned_training_transaction(transaction))
        self.__training_set_y.append(category)

    def __init_training_set(self, training_fname):
        training_set = read_lines(training_fname)
        transactions = [";".join(l.split(';')[:-1]) for l in training_set]
        categories   = [         l.split(';')[-1]   for l in training_set]

        self.__training_set_str = []
        self.__training_set_x = []
        self.__training_set_y = []
        for t,c in zip(transactions, categories):
            self.__enrich_training_set(t, c)

    def __update_state(self):
        self.__categories = {}
        for x,y in zip(self.__corpus, self.__prediction):
            c = self.__categories.setdefault(y, [])
            c.append(x)

        self.__overview = []
        for cat,transacs in self.__categories.items():
            count  = len(transacs)
            amount = sum([float(v.split(';')[-1].replace(' ','')) for v in transacs])
            self.__overview.append((amount, count, cat))
            self.__overview.sort()

    def __predict(self):
        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        self.__text_clf.fit(self.__training_set_x, self.__training_set_y)
        self.__prediction = self.__text_clf.predict(self.__corpus)
        self.__update_state()

    def __init__(self, training_fname, corpus_fnames):
        self.__init_training_set(training_fname)
        # Process all corpus files
        self.__corpus = []
        for fname in corpus_fnames:
            self.__corpus.extend(read_lines(fname))

        self.__vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            token_pattern= '(?u)\\b\\w[a-zA-Z0-9_\\-\\.]+\\b',
            ngram_range=(1,3),
        )
        self.__text_clf = Pipeline([
            ('vect', self.__vectorizer),
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
            fd.write("\n".join(self.__training_set_str))
            fd.write("\n")

    def c_overview(self):
        for amount, count, category in self.__overview:
            print("{:<10} {:<4} {}".format(round(amount,2), count, category))

    def c_list_category(self, category):
        for item in self.__categories[category]:
            # When this becomes too slow, we should use a dictionary to fetch item indexes
            print("{:<5} {}".format(self.__corpus.index(item), item))

    def c_categorize(self, transaction_id, category):
        transaction = self.__corpus[transaction_id]
        self.__enrich_training_set(transaction, category)
        self.__predict()

    def c_debug(self):
        print('\n***** Feature names *****')
        print(self.__vectorizer.get_feature_names())
        print('\n***** Training set str *****')
        print(self.__training_set_str)
        print('\n***** Training set x *****')
        print(self.__training_set_x)
        print('\n***** Training set y *****')
        print(self.__training_set_y)


def help():
    msg="""h                    Display this help message
o                    Display an overview of the expenses / incomes per category
p prediction_fname   Write corpus with predicted categories appended to prediction_fname file
t trainingset_fname  Write training set to trainingset_fname file
l category           List entries in category
c item_id category   Classify item with id item_id to category
q                    Quit"""
    print(msg)

if len(sys.argv) < 3:
    print("Syntax is: {} <training_set> <corpus_1> [<corpus_2> ...]".format(sys.argv[0]))
    exit(1)

training_fname = sys.argv[1]
corpus_fnames  = sys.argv[2:]

corp = Corpus(training_fname, corpus_fnames)
cmd = 'o'
while cmd != 'q':
    try:
      if cmd == '':
          pass
      elif cmd == 'o':
          print("*** Overview")
          corp.c_overview()
      elif cmd == 'p':
          out_fname = ln[1]
          print("*** Writing predictions to '{}'".format(out_fname))
          corp.c_save_prediction(out_fname)
      elif cmd == 'l':
          category = ln[1]
          print("*** Listing items in category '{}'".format(category))
          corp.c_list_category(category)
      elif cmd == 'c':
          item = int(ln[1])
          cat  = ln[2]
          print("*** Classifying '{}' into '{}'".format(item, cat))
          corp.c_categorize(item, cat)
      elif cmd == 't':
          out_fname = ln[1]
          print("*** Saving training file to '{}'".format(out_fname))
          corp.c_save_training(out_fname)
      elif cmd == 'd':
          print("*** Debugging")
          corp.c_debug()
      elif cmd == 'h':
          print("*** Help")
          help()
      else:
          print("*** Unknown command: {}".format(cmd))
          help()
    except Exception as e:
        print("*** Error while processing command '{}'".format(cmd))
        print(traceback.format_exc())
        help()
    try:
        ln = input("> ").split()
        cmd = ln[0]
    except (EOFError, KeyboardInterrupt):
        cmd = 'q'
    except IndexError:
        cmd = ''
