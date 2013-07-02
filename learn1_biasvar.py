#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Use a decision tree classifier to plot overfitting due to allowing too deep a tree"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score
from sklearn import tree
from sklearn import cross_validation
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import unicodecsv


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
    # force any 1.0 (100%) probabilities to be fractionally smaller, so
    # np.log(1-1) doesn't generate a NaN
    probas_class1[np.where(probas_class1 == 1.0)] = 0.999999999999999
    probas_class1[np.where(probas_class1 == 0.0)] = 0.000000000000001
    #import pdb; pdb.set_trace()
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
    #parser.add_argument('--validation_table', help='Table of validation data - get tweets and write predicted class labels back (e.g. learn1_validation_apple)')
    #parser.add_argument('--roc', default=False, action="store_true", help='Plot a Receiver Operating Characterics graph for the learning results')
    #parser.add_argument('--pr', default=False, action="store_true", help='Plot a Precision/Recall graph for the learning results')
    #parser.add_argument('--termmatrix', default=False, action="store_true", help='Draw a 2D matrix of tokens vs binary presence (or absence) using all training documents')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    in_class_lines = reader(in_class_name)
    out_class_lines = reader(out_class_name)

    # put all items into the training set
    train_set = np.array(out_class_lines + in_class_lines)
    target = np.array([0] * len(out_class_lines) + [1] * len(in_class_lines))

    f = plt.figure(1)
    f.clf()
    kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)
    max_branch_depth = range(1, 60, 2)
    all_precisions_train = []
    all_precisions_test = []
    for n in max_branch_depth:
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=2, binary=True, lowercase=True)
        trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
        # get list of feature_names, these occur > 1 time in the dataset
        print("-----------------")
        #print("Feature names (first 20):", vectorizer.get_feature_names()[:20], "...")
        print("Vectorized %d features" % (len(vectorizer.get_feature_names())))

        clf = tree.DecisionTreeClassifier(max_depth=n)

        precisions_train = []
        precisions_test = []
        for i, (train_rows, test_rows) in enumerate(kf):
            Y_train = target[train_rows]
            X_train = trainVectorizerArray[train_rows]
            X_test = trainVectorizerArray[test_rows]
            clf.fit(X_train, Y_train)
            predicted_test = clf.predict(X_test)
            predicted_train = clf.predict(X_train)
            Y_test = target[test_rows]

            precision_train = precision_score(Y_train, predicted_train)
            precision_test = precision_score(Y_test, predicted_test)

            precisions_train.append(precision_train)
            precisions_test.append(precision_test)

        precisions_train = 1 - np.array(precisions_train)
        precisions_test = 1 - np.array(precisions_test)
        test_labels = plt.plot([n] * len(precisions_test), precisions_test, 'og', alpha=0.8, label="Test errors")
        train_labels = plt.plot([n] * len(precisions_train), precisions_train, 'xr', alpha=0.8, label="Training errors")
        plt.draw()

        all_precisions_train.append(np.average(precisions_train))
        all_precisions_test.append(np.average(precisions_test))

    plt.plot(max_branch_depth, all_precisions_test, 'g')
    plt.plot(max_branch_depth, all_precisions_train, 'r')
    plt.xlabel("Decision tree max depth")
    plt.ylabel("Error (1.0-precision)")
    plt.legend((train_labels[0], test_labels[0]), (train_labels[0].get_label(), test_labels[0].get_label()))

    plt.show()
