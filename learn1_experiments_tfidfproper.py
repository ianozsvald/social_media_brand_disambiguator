#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First simple sklearn classifier"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import numpy as np
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
#from sklearn import svm
from sklearn import cross_validation
from nltk.corpus import stopwords
import unicodecsv


############
# NOTE
# this is a basic LogisticRegression classifier, using 5-fold cross validation
# and a cross entropy error measure (which should nicely fit this binary
# decision classification problem).
# do not trust this code to do anything useful in the real world!
#
# this code creates a TF-IDF model inside a cross validation loop
############


def reader(class_name):
    class_reader = unicodecsv.reader(open(class_name), encoding='utf-8')
    row0 = next(class_reader)
    assert row0 == ["tweet_id", "tweet_text"]
    lines = []
    for tweet_id, tweet_text in class_reader:
        txt = tweet_text.strip()
        if len(txt) > 0:
            lines.append(txt)
    return lines


def cross_entropy_error(Y, probas_):
    # compute Cross Entropy using the Natural Log:
    # ( -tln(y) ) − ( (1−t)ln(1−y) )
    probas_class1 = probas_[:, 1]  # get the class 1 probabilities
    cross_entropy_errors = ((-Y) * (np.log(probas_class1))) - ((1 - Y) * (np.log(1 - probas_class1)))
    return cross_entropy_errors


def show_errors(cross_entropy_errors_by_fold, method="cross entropy", lower_is_better=True):
    print("Cross validation %s errors:" % (method) + str(cross_entropy_errors_by_fold))
    if lower_is_better:
        note = "(lower is better)"
    else:
        note = "(higher is better)"

    print("%s %s: %0.2f (+/- %0.2f)" % (method, note, cross_entropy_errors_by_fold.mean(), cross_entropy_errors_by_fold.std() / 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple sklearn implementation, example usage "learn1.py scikit_testtrain_apple --validation_table=learn1_validation_apple"')
    parser.add_argument('table', help='Name of in and out of class data to read (e.g. scikit_validation_app)')
    parser.add_argument('--validation_table', help='Table of validation data - get tweets and write predicted class labels back (e.g. learn1_validation_apple)')
    parser.add_argument('--roc', default=False, action="store_true", help='Plot a Receiver Operating Characterics graph for the learning results')
    parser.add_argument('--pr', default=False, action="store_true", help='Plot a Precision/Recall graph for the learning results')
    parser.add_argument('--termmatrix', default=False, action="store_true", help='Draw a 2D matrix of tokens vs binary presence (or absence) using all training documents')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    in_class_lines = reader(in_class_name)
    out_class_lines = reader(out_class_name)

    # put all items into the training set
    train_set = np.array(out_class_lines + in_class_lines)
    target = np.array([0] * len(out_class_lines) + [1] * len(in_class_lines))

    # choose a vectorizer to turn the tokens in tweets into a matrix of
    # examples (we can plot this further below using --termmatrix)
    stopWords = stopwords.words('english')
    MIN_DF = 2
    vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=(1, 3))
    #vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=(1, 2))
    #vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=(1, 3))
    vectorizer_tfidf = TfidfVectorizer(stop_words=stopWords, min_df=MIN_DF, ngram_range=(1, 3))#, sublinear_tf=True)
    vectorizer = vectorizer_tfidf
    #vectorizer = vectorizer_binary
    print(vectorizer)

    clf = linear_model.LogisticRegression()
    #clf = svm.LinearSVC()
    #clf = linear_model.LogisticRegression(penalty='l2', C=1.2)

    kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)

    # try the idea of calculating a cross entropy score per fold
    cross_entropy_errors_test_by_fold = np.zeros(len(kf))
    cross_entropy_errors_train_by_fold = np.zeros(len(kf))
    precisions_by_fold = np.zeros(len(kf))
    for i, (train_rows, test_rows) in enumerate(kf):
        tweets_train_rows = train_set[train_rows]  # select training rows
        tweets_test_rows = train_set[test_rows]  # select testing rows
        Y_train = target[train_rows]
        Y_test = target[test_rows]
        X_train = vectorizer.fit_transform(tweets_train_rows).toarray()
        X_test = vectorizer.transform(tweets_test_rows).todense()

        clf.fit(X_train, Y_train)
        probas_test_ = clf.predict_proba(X_test)
        probas_train_ = clf.predict_proba(X_train)
        # compute cross entropy for all trained and tested items in this fold
        cross_entropy_errors_test = cross_entropy_error(Y_test, probas_test_)
        cross_entropy_errors_train = cross_entropy_error(Y_train, probas_train_)
        cross_entropy_errors_test_by_fold[i] = np.average(cross_entropy_errors_test)
        cross_entropy_errors_train_by_fold[i] = np.average(cross_entropy_errors_train)
        precisions_by_fold[i] = precision_score(Y_test, clf.predict(X_test))

    print("Training:")
    show_errors(cross_entropy_errors_train_by_fold)
    print("Testing:")
    show_errors(cross_entropy_errors_test_by_fold)
    print("Precisions:")
    show_errors(precisions_by_fold, method="precision", lower_is_better=False)
