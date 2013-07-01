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
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import cross_validation
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import unicodecsv
import sql_convenience

# NOTE
# this biar/var version may not yet do the right job, I must revisit it

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


def cross_entropy_error(Y, probas_):
    # compute Cross Entropy using the Natural Log:
    # ( -tln(y) ) − ( (1−t)ln(1−y) )
    probas_class1 = probas_[:, 1]  # get the class 1 probabilities
    cross_entropy_errors = ((-Y) * (np.log(probas_class1))) - ((1 - Y) * (np.log(1 - probas_class1)))
    return cross_entropy_errors


def show_cross_validation_errors(cross_entropy_errors_by_fold):
    print("Cross validation cross entropy errors:" + str(cross_entropy_errors_by_fold))
    print("Cross entropy (lower is better): %0.3f (+/- %0.3f)" % (cross_entropy_errors_by_fold.mean(), cross_entropy_errors_by_fold.std() / 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple sklearn implementation, example usage "learn1.py scikit_testtrain_apple --validation_table=learn1_validation_apple"')
    parser.add_argument('table', help='Name of in and out of class data to read (e.g. scikit_validation_app)')
    parser.add_argument('--validation_table', help='Table of validation data - get tweets and write predicted class labels back (e.g. learn1_validation_apple)')
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
    #MIN_DF = 2
    #vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True)
    #vectorizer_tfidf = TfidfVectorizer(stop_words=stopWords, min_df=MIN_DF)
    #vectorizer = vectorizer_tfidf
    #vectorizer = vectorizer_binary

    f = plt.figure(1)
    f.clf()
    for n in np.arange(1, 10):
        vectorizer = CountVectorizer(stop_words=stopWords, ngram_range=(1, 2), min_df=n, binary=True, lowercase=True)
        #vectorizer = CountVectorizer(stop_words=stopWords, ngram_range=(1, n), min_df=3, binary=True, lowercase=True)
        trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
        # get list of feature_names, these occur > 1 time in the dataset
        print("-----------------")
        print("Feature names (first 20):", vectorizer.get_feature_names()[:20], "...")
        print("Vectorized %d features" % (len(vectorizer.get_feature_names())))

        #clf = linear_model.LogisticRegression(penalty='l1', C=1)
        #clf = linear_model.LogisticRegression(penalty='l2', C=10)
        clf = linear_model.LogisticRegression()

        kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)
        # using a score isn't so helpful here (I think) as I want to know the
        # distance from the desired categories and a >0.5 threshold isn't
        # necessaryily the right thing to measure (I care about precision when
        # classifying, not recall, so the threshold matters)
        #cross_val_scores = cross_validation.cross_val_score(clf, trainVectorizerArray, target, cv=kf, n_jobs=-1)
        #print("Cross validation in/out of class test scores:" + str(cross_val_scores))
        #print("Accuracy: %0.3f (+/- %0.3f)" % (cross_val_scores.mean(), cross_val_scores.std() / 2))

        # try the idea of calculating a cross entropy score per fold
        cross_entropy_errors_test_by_fold = np.zeros(len(kf))
        cross_entropy_errors_train_by_fold = np.zeros(len(kf))
        for i, (train_rows, test_rows) in enumerate(kf):
            Y_train = target[train_rows]
            X_train = trainVectorizerArray[train_rows]
            X_test = trainVectorizerArray[test_rows]
            probas_test_ = clf.fit(X_train, Y_train).predict_proba(X_test)
            probas_train_ = clf.fit(X_train, Y_train).predict_proba(X_train)
            # compute cross entropy for all trained and tested items in this fold
            Y_test = target[test_rows]

            cross_entropy_errors_test = cross_entropy_error(Y_test, probas_test_)
            cross_entropy_errors_train = cross_entropy_error(Y_train, probas_train_)
            cross_entropy_errors_test_by_fold[i] = np.average(cross_entropy_errors_test)
            cross_entropy_errors_train_by_fold[i] = np.average(cross_entropy_errors_train)
        #import pdb; pdb.set_trace()
        print("Training:")
        show_cross_validation_errors(cross_entropy_errors_train_by_fold)
        print("Testing:")
        show_cross_validation_errors(cross_entropy_errors_test_by_fold)

        nbr_features = trainVectorizerArray.shape[1]
        x_value = nbr_features
        #x_value = n
        plt.plot([x_value] * len(cross_entropy_errors_test), cross_entropy_errors_test, 'og', alpha=0.8)
        plt.plot([x_value] * len(cross_entropy_errors_train), cross_entropy_errors_train, 'xr', alpha=0.8)

    # write validation results to specified table
    if args.validation_table:
        # make sparse training set using all of the test/train data (combined into
        # one set)
        train_set_sparse = vectorizer.transform(train_set)
        # instantiate a local classifier
        clfl = clf.fit(train_set_sparse.todense(), target)

        # check and print out two classifications as sanity checks
        check_classification(vectorizer, clfl)
        # use a threshold (arbitrarily chosen at present), test against the
        # validation set and write classifications to DB for reporting
        chosen_threshold = 0.92
        label_learned_set(vectorizer, clfl, chosen_threshold, args.validation_table)
