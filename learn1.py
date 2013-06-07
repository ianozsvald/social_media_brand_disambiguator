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
from sklearn import linear_model
#from sklearn import svm
from sklearn import cross_validation
from nltk.corpus import stopwords
import unicodecsv


############
# NOTE
# this is HACKY CODE just to get something in place (end-to-end from making
# gold standard to getting a simple ML classifier working)
# do not trust this code to do anything useful in the real world!
############


def reader(class_name):
    class_reader = unicodecsv.reader(open(class_name), encoding='utf-8')
    row0 = next(class_reader)
    assert row0 == ["tweet_id", "tweet_text"]
    #class_writer.writerow(("tweet_id", "tweet_text"))
    lines = []
    for tweet_id, tweet_text in class_reader:
        txt = tweet_text.strip()
        if len(txt) > 0:
            lines.append(txt)
    return lines

#def read_data(filename):
    #f = open(filename)
    #lines = f.readlines()
    #lines = [unicode(line.strip(), 'utf-8') for line in lines if len(line.strip()) > 0]
    #return lines


#def tokenize(items):
    #"""Create list of >1 char length tokens, split by punctuation"""
    #tokenised = []
    #for tweet in items:
        #tokens = nltk.tokenize.WordPunctTokenizer().tokenize(tweet)
        #tokens = [token for token in tokens if len(token) > 1]
        #tokenised.append(tokens)
    #return tokenised


#def clean_tweet(tweet, tweet_parser):
    #tweet_parser.describe_tweet(tweet)
    #components = tweet_parser.get_components()

    #filtered_tweet = " ".join(tweet_parser.get_tokens(filtered_components))
    #return filtered_tweet


def check_classification(vectorizer, clfl):
    spd0 = vectorizer.transform([u'really enjoying how the apple\'s iphone makes my ipad look small']).todense()
    print("1?", clfl.predict(spd0), clfl.predict_proba(spd0))  # -> 1 which is set 1 (is brand)
    spd1 = vectorizer.transform([u'i like my apple, eating it makes me happy']).todense()
    print("0?", clfl.predict(spd1), clfl.predict_proba(spd1))  # -> 0 which is set 0 (not brand)
    # i can use vectorizer.get_feature_names()[spd_item] having tried
    # np.where(spdx>0) to get a list of item positions to see what the features
    # are

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple sklearn implementation')
    parser.add_argument('table', help='Name of in and out of class data to read (e.g. annotations_apple)')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    in_class_lines = reader(in_class_name)
    out_class_lines = reader(out_class_name)

    # put all items into the training set
    train_set = in_class_lines + out_class_lines
    target = np.array([1] * len(in_class_lines) + [0] * len(out_class_lines))

    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=stopWords, ngram_range=(1, 1), min_df=1)
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    # get list of feature_names, these occur > 1 time in the dataset
    print("Feature names:", vectorizer.get_feature_names()[:20], "...")
    print("Found %d features" % (len(vectorizer.get_feature_names())))

    clf_logreg = linear_model.LogisticRegression()  # C=1e5)
    clf = clf_logreg
    #clf_svc = svm.SVC(kernel='linear', probability=True)
    #clf = clf_svc

    kf = cross_validation.LeaveOneOut(n=len(target))  # KFold(n=len(target), k=10, shuffle=True)
    print("Shortcut cross_val_score to do the same thing, using all CPUs:")
    #t1 = datetime.datetime.now()
    cross_val_scores = cross_validation.cross_val_score(clf, trainVectorizerArray, target, cv=kf, n_jobs=-1)
    #dt1 = datetime.datetime.now() - t1
    print(np.average(cross_val_scores))
    #print("Time taken:", dt1)

    # make sparse training set, 89x586 elements
    train_set_sparse = vectorizer.transform(train_set)
    # instantiate a local classifier
    clfl = clf.fit(train_set_sparse.todense(), target)

    check_classification(vectorizer, clfl)
