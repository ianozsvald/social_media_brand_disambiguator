#!/usr/bin/env python
"""1 liner to explain this project"""
# -*- coding: utf-8 -*-
import ujson as json
import logging
from dateutil import parser as dt_parser


def get_tweets(tweets):
    """Generator to return entry from valid JSON lines"""
    for tweet in tweets:
        # load with json to validate
        try:
            tw = json.loads(tweet)
            yield tw
        except ValueError as err:
            logging.debug("Odd! We have a ValueError when json.loads(tweet): %r" % repr(err))


#def filter_http(tweets):
    #"""Ignore links with http links (can be useful to ignore spam)"""
    #for tweet in tweets:
        #try:
            #if 'http' not in tweet['text']:
                #yield tweet
        #except KeyError as err:
            #logging.debug("Odd! We have a KeyError: %r" % repr(err))


def get_tweet_body(tweets):
    """Get tweets, ignore ReTweets"""
    for tweet in tweets:
        try:
            if 'text' in tweet:
                if not tweet['text'].startswith('RT'):
                    tweet['created_at'] = dt_parser.parse(tweet['created_at'])
                    yield tweet
        except KeyError as err:
            logging.debug("Odd! We have a KeyError: %r" % repr(err))


def files(file_list):
    """Yield lines from a list of input json data files"""
    for filename in file_list:
        f = open(filename)
        for line in f:
            yield line
