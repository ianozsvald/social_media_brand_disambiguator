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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import cross_validation
#from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt
#from matplotlib import cm
from nltk.corpus import stopwords
import unicodecsv


############
# NOTE
# this is a basic LogisticRegression classifier, using 5-fold cross validation
# and a cross entropy error measure (which should nicely fit this binary
# decision classification problem).
# do not trust this code to do anything useful in the real world!
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

    # choose a vectorizer to turn the tokens in tweets into a matrix of
    # examples (we can plot this further below using --termmatrix)
    stopWords = stopwords.words('english')
    MIN_DF = 2
    NGRAM_RANGE = (1, 2)
    vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=NGRAM_RANGE)

    #vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=(1, 2))
    #vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True, ngram_range=(1, 3))
    vectorizer_tfidf = TfidfVectorizer(stop_words=stopWords, min_df=MIN_DF, ngram_range=NGRAM_RANGE)
    #vectorizer = vectorizer_tfidf
    vectorizer = vectorizer_binary
    print(vectorizer)

    #clf = linear_model.LogisticRegression(penalty='l2', C=1.2)
    _ = linear_model.LogisticRegression()
    _ = svm.LinearSVC()
    _ = naive_bayes.BernoulliNB()  # useful for binary inputs (MultinomialNB is useful for counts)
    _ = naive_bayes.GaussianNB()
    _ = naive_bayes.MultinomialNB()
    _ = ensemble.AdaBoostClassifier(n_estimators=100, base_estimator=tree.DecisionTreeClassifier(max_depth=2, criterion='entropy'))
    #clf = ensemble.AdaBoostClassifier(n_estimators=100, base_estimator=tree.DecisionTreeClassifier(max_depth=2))
    _ = tree.DecisionTreeClassifier(max_depth=50, min_samples_leaf=5)
    #clf = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, criterion='entropy')
    #clf = ensemble.RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=10, oob_score=False, n_jobs=-1, criterion='entropy')
    _ = ensemble.RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_estimators=50, n_jobs=-1, criterion='entropy')
    #clf = ensemble.RandomForestClassifier(max_depth=30, min_samples_leaf=5, n_estimators=100, oob_score=True, n_jobs=-1)
    clf = neighbors.KNeighborsClassifier(n_neighbors=11)

    print(clf)

    kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)

    f = plt.figure(1)
    f.clf()

    # try the idea of calculating a cross entropy score per fold
    cross_entropy_errors_test_by_fold = np.zeros(len(kf))
    cross_entropy_errors_train_by_fold = np.zeros(len(kf))

    precisions_by_fold = np.zeros(len(kf))
    # build arrays of all the class 0 and 1 probabilities (matching the class 0
    # and 1 gold tags)
    probabilities_class_0_Y_test_all_folds = np.array([])
    probabilities_class_1_Y_test_all_folds = np.array([])
    # list of all the false positives for later diagnostic
    all_false_positives_zipped = []
    for i, (train_rows, test_rows) in enumerate(kf):
        tweets_train_rows = train_set[train_rows]  # select training rows
        tweets_test_rows = train_set[test_rows]  # select testing rows
        Y_train = target[train_rows]
        Y_test = target[test_rows]
        X_train = vectorizer.fit_transform(tweets_train_rows).toarray()
        X_test = vectorizer.transform(tweets_test_rows).todense()

        clf.fit(X_train, Y_train)
        probas_test_ = clf.predict_proba(X_test)

        predictions_test = clf.predict(X_test)
        # figure out false positive rows from X_test (which is a subset from
        # train_set)
        false_positive_locations = np.where(Y_test - predictions_test == -1)  # 0 (truth) - 1 (prediction) == -1 which is a false positive
        false_positive_tweet_rows = test_rows[np.where(Y_test - predictions_test == -1)]
        false_positive_tweets = train_set[false_positive_tweet_rows]
        bag_of_words_false_positive_tweets = vectorizer.inverse_transform(X_test[false_positive_locations])
        false_positives_zipped = zip(false_positive_tweets, bag_of_words_false_positive_tweets)
        all_false_positives_zipped.append(false_positives_zipped)
        #import pdb; pdb.set_trace()

        # select and concatenate the class 0 and 1 probabilities to their
        # respective arrays for later investigation
        probabilities_class_1_Y_test = probas_test_[np.where(Y_test == 1)]  # get all probabilities for class 1
        nbr_features_X_test = [np.sum(row) for row in X_test[np.where(Y_test == 1)]]
        class_1_labels = plt.scatter(nbr_features_X_test, probabilities_class_1_Y_test[:, 1], c='c', edgecolor='none', label="Class 1")
        probabilities_class_0_Y_test = probas_test_[np.where(Y_test == 0)]  # get all probabilities for class 0
        nbr_features_X_test = [np.sum(row) for row in X_test[np.where(Y_test == 0)]]
        class_0_labels = plt.scatter(nbr_features_X_test, probabilities_class_0_Y_test[:, 1], c='k', edgecolor='none', label="Class 0")

        probas_train_ = clf.predict_proba(X_train)
        # compute cross entropy for all trained and tested items in this fold
        if True:
            cross_entropy_errors_test = cross_entropy_error(Y_test, probas_test_)
            cross_entropy_errors_train = cross_entropy_error(Y_train, probas_train_)
            cross_entropy_errors_test_by_fold[i] = np.average(cross_entropy_errors_test)
            cross_entropy_errors_train_by_fold[i] = np.average(cross_entropy_errors_train)
        precisions_by_fold[i] = precision_score(Y_test, clf.predict(X_test))

        print(len(test_rows))
        probabilities_class_0_Y_test_all_folds = np.concatenate((probabilities_class_0_Y_test_all_folds, probabilities_class_0_Y_test[:, 1]))
        probabilities_class_1_Y_test_all_folds = np.concatenate((probabilities_class_1_Y_test_all_folds, probabilities_class_1_Y_test[:, 1]))

    plt.legend((class_1_labels, class_0_labels), (class_1_labels.get_label(), class_0_labels.get_label()), scatterpoints=2, loc=7)
    plt.xlim(xmin=-1)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Number of features for example')
    plt.ylabel('Probability of class 1 for example')
    plt.title("{} class probabilities with {} features".format(str(clf.__class__).split('.')[-1][:-2], len(vectorizer.get_feature_names())))
    plt.show()

    # trial of grid search
    #from sklearn.grid_search import GridSearchCV
    #grid_search = GridSearchCV(linear_model.LogisticRegression(), {'C': np.power(10.0, np.arange(-3, 3, step=0.5))}, n_jobs=-1, verbose=1)
    #res=grid_search.fit(X_train, Y_train)
    #res.best_params_

    if isinstance(clf, tree.DecisionTreeClassifier):
        # print the most important features
        feature_importances = zip(clf.feature_importances_, vectorizer.get_feature_names())
        feature_importances.sort(reverse=True)
        print("Most important features:", feature_importances[:10])

        with open("dectree.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f, feature_names=vectorizer.get_feature_names())
        os.system("dot -Tpdf dectree.dot -o dectree.pdf")  # turn dot into PDF for visual
            # diagnosis

    print("Training:")
    show_errors(cross_entropy_errors_train_by_fold)
    print("Testing:")
    show_errors(cross_entropy_errors_test_by_fold)
    print("Precisions:")
    show_errors(precisions_by_fold, method="precision", lower_is_better=False)
