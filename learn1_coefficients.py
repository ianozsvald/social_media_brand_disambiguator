#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First simple sklearn classifier"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import copy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import cross_validation
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import unicodecsv
import sql_convenience


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


def label_learned_set(vectorizer, clfl, threshold, validation_table):
    for row in sql_convenience.extract_classifications_and_tweets(validation_table):
        cls, tweet_id, tweet_text = row
        spd = vectorizer.transform([tweet_text]).todense()
        predicted_cls = clfl.predict(spd)
        predicted_class = predicted_cls[0]  # turn 1D array of 1 item into 1 item
        predicted_proba = clfl.predict_proba(spd)[0][predicted_class]
        if predicted_proba < threshold and predicted_class == 1:
            predicted_class = 0  # force to out-of-class if we don't trust our answer
        sql_convenience.update_class(tweet_id, validation_table, predicted_class)


def check_classification(vectorizer, clfl):
    spd0 = vectorizer.transform([u'really enjoying how the apple\'s iphone makes my ipad look small']).todense()
    print("1?", clfl.predict(spd0), clfl.predict_proba(spd0))  # -> 1 which is set 1 (is brand)
    spd1 = vectorizer.transform([u'i like my apple, eating it makes me happy']).todense()
    print("0?", clfl.predict(spd1), clfl.predict_proba(spd1))  # -> 0 which is set 0 (not brand)


def annotate_tokens(indices_for_large_coefficients, clf, vectorizer, plt):
    y = clf.coef_[0][indices_for_large_coefficients]
    tokens = np.array(vectorizer.get_feature_names())[indices_for_large_coefficients]
    for x, y, token in zip(indices_for_large_coefficients, y, tokens):
        plt.text(x, y, token)


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
    train_set = out_class_lines + in_class_lines
    target = np.array([0] * len(out_class_lines) + [1] * len(in_class_lines))

    # choose a vectorizer to turn the tokens in tweets into a matrix of
    # examples (we can plot this further below using --termmatrix)
    stopWords = stopwords.words('english')
    MIN_DF = 2
    vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True)
    vectorizer_tfidf = TfidfVectorizer(stop_words=stopWords, min_df=MIN_DF)
    #vectorizer = vectorizer_tfidf
    vectorizer = vectorizer_binary

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    print("Feature names (first 20):", vectorizer.get_feature_names()[:20], "...")
    print("Vectorized %d features" % (len(vectorizer.get_feature_names())))

    MAX_PLOTS = 3
    f = plt.figure(1)
    plt.clf()
    f = plt.subplot(MAX_PLOTS, 1, 0)

    for n in range(MAX_PLOTS):
        if n == 0:
            clf = naive_bayes.BernoulliNB()
            title = "Bernoulli Naive Bayes"
        if n == 1:
            clf = linear_model.LogisticRegression()
            title = "Logistic Regression l2 error"
        if n == 2:
            clf = linear_model.LogisticRegression(penalty='l1')
            title = "Logistic Regression l1 error"

        kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)
        # using a score isn't so helpful here (I think) as I want to know the
        # distance from the desired categories and a >0.5 threshold isn't
        # necessaryily the right thing to measure (I care about precision when
        # classifying, not recall, so the threshold matters)
        #cross_val_scores = cross_validation.cross_val_score(clf, trainVectorizerArray, target, cv=kf, n_jobs=-1)
        #print("Cross validation in/out of class test scores:" + str(cross_val_scores))
        #print("Accuracy: %0.3f (+/- %0.3f)" % (cross_val_scores.mean(), cross_val_scores.std() / 2))

        f = plt.subplot(MAX_PLOTS, 1, n + 1)
        plt.title(title)

        for i, (train_rows, test_rows) in enumerate(kf):
            Y_train = target[train_rows]
            X_train = trainVectorizerArray[train_rows]
            X_test = trainVectorizerArray[test_rows]
            probas_test_ = clf.fit(X_train, Y_train).predict_proba(X_test)
            probas_train_ = clf.fit(X_train, Y_train).predict_proba(X_train)

            # plot the Logistic Regression coefficients

            if n == 1 or n == 2:
                coef = clf.coef_[0]
            if n == 0:
                coef = clf.coef_
            plt.plot(coef, 'b', alpha=0.3)
            plt.ylabel("Coefficient value")
        xmax = coef.shape[0]
        plt.xlim(xmax=xmax)

    plt.xlabel("Coefficient per term")
    # plot the tokens with the largest coefficients
    coef = copy.copy(clf.coef_[0])
    coef.sort()
    annotate_tokens(np.where(clf.coef_ >= coef[-10])[1], clf, vectorizer, plt)
    annotate_tokens(np.where(clf.coef_ < coef[10])[1], clf, vectorizer, plt)

    #f = plt.subplot(MAX_PLOTS, 1, 1)
    #plt.title("{}: l2 penalty (top) vs l1 penalty (bottom) for {} cross fold models on {}".format(str(clf.__class__).split('.')[-1][:-2], len(kf), args.table))
    plt.show()
