#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Configuration provided by 'import config' and DISAMBIGUATOR_CONFIG env var"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import os
import sqlite3
import logging

# Read DISAMBIGUATOR_CONFIG environment variable (raise error if missing or badly
# configured), use this to decide on our config and import the relevant python
# file

# This assumes that locally we have suitable python files e.g. production.py,
# testing.py
CONFIG_ENV_VAR = "DISAMBIGUATOR_CONFIG"
CONFIG_ENV_VAR_PRODUCTION = "production"
CONFIG_ENV_VAR_TESTING = "testing"
config_set = False  # only set to True if we have find a valid ENV VAR
config_choice = os.getenv(CONFIG_ENV_VAR)
# we could use testing by default, if we choose to
if config_choice is None:
    print("Defaulting in {} to environment: {} as env var {} was not specified.".format(__file__, CONFIG_ENV_VAR_TESTING, CONFIG_ENV_VAR))
    config_choice = CONFIG_ENV_VAR_TESTING

if config_choice == CONFIG_ENV_VAR_PRODUCTION:
    from .production import *
    config_set = True
if config_choice == CONFIG_ENV_VAR_TESTING:
    from .testing import *
    config_set = True
if not config_set:
    raise ValueError("ALERT! ENV VAR \"{}\" must be set e.g. \"export {}={}\"".format(CONFIG_ENV_VAR, CONFIG_ENV_VAR, CONFIG_ENV_VAR_TESTING))

# Simple logging configuration, an example output might be:
# 2013-06-03 15:07:55.740 p7470 {start_here.py:31} INFO - This is an example log message
LOG_FILE_NAME = "log.log"
# The date format is ISO 8601, format includes a decimal separator for
# milliseconds (not the default comma) as dateutil.parser cannot read the
# command but it can read the decimal separator (both are allowed in ISO 8601)
logging.basicConfig(filename=LOG_FILE_NAME, level=logging.DEBUG, format='%(asctime)s.%(msecs)d p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# note that it might be useful to use the ConcurrentLogHandler or
# RotatingLogHandler here (either require some more setup)

# make sqlite db, add default
db_conn = sqlite3.connect(sqldb)
db_conn.row_factory = sqlite3.Row  # use Row to return dict-like results
