 social_media_brand_disambiguator
=================================

Brand disambiguator for tweets to differentiate e.g. Orange vs orange (brand vs foodstuff), using NLTK and scikit-learn

NOTE this is a work in progress, started June 2013, currently it doesn't do very much at all

NOTE NOTE NOTE ! this checkin (10th June 2013) gets my working code checked-in before two talks. I need to go back and do some refactoring (removing some hardcoded table names, improving these docs) before this will work smoothly. YOU HAVE BEEN WARNED.

I'm only using English tweets and only annotating the ones I can clearly differentiate.

Write-ups
---------

There are some write-ups and presentations online if you'd like some background:

  * http://ianozsvald.com/category/socialmediabranddisambiguator/
  * https://speakerdeck.com/ianozsvald/detecting-the-right-apples-and-oranges-1-hour-talk-on-python-for-brand-disambiguation-using-scikit-learn-at-brightonpython-june-2013  # June 2013 at BrightonPython and DataScienceLondon

Setup
-----

The code runs with Python 2.7 using sqlite.

    $ git clone https://github.com/ianozsvald/social_media_brand_disambiguator.git ./src  # clones code into local ./src folder
    $ virtualenv ./env  # do this in the same folder, not the src subfolder (it would be ok but it is cleaner to leave the environment separate from the src)
    $ . env/bin/activate  # '.' is short for 'source', it runs the env/bin/activate script to enable the virtualenv
    $ # note that our path line probably changes to show '(env)' as we have activated the virtualenv
    $ cd src
    $ pip install -r requirements.txt  # base libs, installed into our virtualenv
    $ ipython  # now we'll test the basic installation
    $   In [1]: import numpy  # test that numpy can be imported
    $   In [2]: import pandas  # test that pandas can be imported
    $ # oddly the cld won't be installed via requirements.txt as the path is too long (see below), so we do it by hand:
    $ pip install chromium-compact-language-detector==0.2  # language detector for annotating
    $ ipython  # test that cld is installed
    $   In [1]: import cld
    $ pip install -r requirements_2.txt  # get around MEP11/sklearn requirement for numpy with second requirements file
    $ ipython  # another quick test
    $   In [1]: import sklearn
    $   In [2]: import matplotlib

Note that if you get any errors, you might be missing required dependencies in your current setup. E.g. on linux I install `numpy` using the Ubuntu package manager (apt-get) so that the core dependencies (like ATLAS) are pre-installed, then I install `numpy` using `pip` to the named version (and it uses the OS-installed dependencies that came in via apt-get). You'll have to read up on what's required for your OS depending on which library breaks.

Note that the `pip` `IOError: [Errno 36 File name too long` error is related to my Ubuntu installation with an encrypted home directory as detailed here: https://bugs.launchpad.net/ecryptfs/+bug/344878

Note that generally I'm using Python 3 `__future__` imports, the code isn't tested with Python 3 but the porting should be straight-forward. sqlite only wants byte/strings for key indexing (not unicode strings).

Tests
-----

If you're in the `src\` folder and you've activated your virtualenv then you should be ready to run:

    $ nosetests  # the project defaults to 'testing' if DISAMBIGUATOR_CONFIG isn't set
    $ nosetests -s  # runs without capturing stdout, useful if you're using `import pdb; pdb.set_trace()` for debugging
    $ nosetests --with-coverage --cover-html  # with an HTML coverage report to cover/index.html

Note if you see `ImportError: cannot import name opencalais_key` then if you plan to use OpenCalais, you need an API key (but if you don't plan to use their NER then you don't need the key and you can ignore this test failure).

A word on environment variables
-------------------------------

I'm running on Linux, I use an environment variable `DISAMBIGUATOR_CONFIG` to tell the code if we're running in `testing` or `production`. This switches between an in-memory SQLite DB for testing which is blanked and an on-disk SQLite DB which is not blanked. By default if the environment variable has not been set, the code will assume that we're in `testing` mode (so nothing is stored beyond the single session). 

If you want to use SQLite DB then the code expects to see something like `data/annotations.sqlite`. You can then either put `DISAMBIGUATOR_CONFIG=production` before whatever you run, or `export DISAMBIGUATOR_CONFIG=production` for that shell window (but if you then run the `nosetests` remember that it'll think it is in `production` mode and will start to mess with the production database).

A word on SQLite Databases
--------------------------

SQLite is built into Python 2.7, it is a lightweight single-file database. You can investigate it at the command line usinng `sqlite3`. In Firefox look at https://addons.mozilla.org/en-us/firefox/addon/sqlite-manager/ for a nice graphical manager.

Creating a gold standard
------------------------

If you have a SQLiteDB then you won't need to do this.

    $ u'/home/ian/workspace/virtualenvs/tweet_disambiguation_project/prototype1/src'
    $ #export DISAMBIGUATOR_CONFIG=production  # might be useful if not using ipython
    $ DISAMBIGUATOR_CONFIG=production ipython
    $ %run tweet_annotator.py ../../apple10.json apple
    # or
    $ DISAMBIGUATOR_CONFIG=production python tweet_annotator.py ../../apple10.json apple


Annotating the tweets using OpenCalais
--------------------------------------

OpenCalais have a strong named entity recognition API offering, we can use it to annotate tweets to see how it does. You need an API key from them via http://www.opencalais.com/APIkey and you need to copy the key into a 1 line file named `ner_apis/opencalais/opencalais_key.py` which will look like:

    $ API_KEY = "<opencalais-key>"

You can run the annotator using:

    $ DISAMBIGUATOR_CONFIG=production python ner_annotator.py apple opencalais --drop  # optionally drop the destination table so we start afresh
    $ DISAMBIGUATOR_CONFIG=production python ner_annotator.py apple opencalais # run in another window to double fetching speed


Creating a test/train and validation subset
-------------------------------------------

Copy a subset of the annotations table to a test/train set and a validation set:

Testing:

    $ DISAMBIGUATOR_CONFIG=production python make_db_subset.py annotations_apple 10 scikit_testtrain_apple 5 scikit_validation_apple --drop

Real:

    $ DISAMBIGUATOR_CONFIG=production python make_db_subset.py annotations_apple 584 scikit_testtrain_apple 100 scikit_validation_apple --drop

Exporting results
-----------------

Output an ordered list of classifications and tweets (by tweet_id), allows a diff (e.g. using meld):

    $ DISAMBIGUATOR_CONFIG=production python export_classified_tweets.py annotations_apple > output/annotations_apple.csv
    $ DISAMBIGUATOR_CONFIG=production python export_classified_tweets.py opencalais_apple > output/opencalais_apple.csv

    $ DISAMBIGUATOR_CONFIG=production python score_results.py annotations_apple opencalais_apple

Exporting test/train data
-------------------------

    $ DISAMBIGUATOR_CONFIG=production python export_inclass_outclass.py annotations_apple

Simple learning
---------------

Use sklearn in a trivial way to classify in and out of class examples. learn1 uses Leave One Out Cross Validation with a Logistic Regression classifier using default text preparation methods, the results are pretty poor. Note that there is no real validation set (just an out of sample test for the 2 cases as a sanity check after training). This is a trivial classifier and isn't to be trusted for any real work.

Results are written to the table specified as the second argument:

******* NOTE use SQLiteManager to copy scikit_validation_app to learn1_validation_apple

    $ DISAMBIGUATOR_CONFIG=production python learn1.py scikit_testtrain_apple --validation_table=learn1_validation_apple

Scoring predictions
-------------------

We can score another table (e.g. predicitions from the scikit code - forthcoming) against the gold standard, it outputs Precision and Recall.

    $ DISAMBIGUATOR_CONFIG=production python score_results.py annotations_apple scikit_apple

To compare the validation subset of a Gold Standard to e.g. the OpenCalais equivalent use:

    $ DISAMBIGUATOR_CONFIG=production python score_results.py scikit_validation_apple opencalais_apple
    $ DISAMBIGUATOR_CONFIG=production python score_results.py scikit_validation_apple learn1_validation_apple

TODO
----

Here are a few ideas of mini projects if you'd like to collaborate but aren't sure about the machine-learning side of things:

  * IAN - export a list of tweet ids and class labels to a .csv file
  * Take my list of tweet-ids, fetch them from Twitter and insert into SQLite using sql_convenience, add the classification that I'll also provide (note - ask me for this list as it isn't in the repo yet)
  * Inherit ner_api_caller.py and build an interface to DBPediaSpotlight, include fixtures in the test code
  * Extract a validation set of 100 in- and 100 out-of-class tweets, send to DBPedia
  * Extend ner_api_caller.py and build a local simple "capitalised brand detector" (e.g. it only looks for the exact term "Apple" in the tweet) - it'll do a similar job to the OpenCalais and DBPedia API calls but without an API call as the code will be local (and really simple) - this would be a very sensible baseline tool
  * Extend ner_api_caller.py and build a heuristic based classifier - e.g. maybe look at capitalised first letter in the target word and a couple of obvious apple-related terms (e.g. iphone, phone, ios, ipad etc) - this would be the equivalent to a human writing some Boolean rules and it serves as another sensible baseline (if we can't beat this then it raises the question "does this system offer anything that a human couldn't already do?")

Design flaws
------------

  * sqlite table structure assumes 1 brand per table (e.g. annotations_apple with class set to is-apple-brand or is-apple-somethingelse), this isn't normal form but is probably fine for the prototype

Other notes
-----------
  
  * https://github.com/twitter/twitter-text-rb/blob/master/lib/twitter-text/regex.rb  Notes from twitter on how they handle unicode:
  * http://nerd.eurecom.fr/documentation  possibly worth considering these other APIs?


3rd party bugs
--------------

SQLite Manager (0.8) in Firefox suffers from a large-int rounding bug due to JavaScript, large ints like Twitter Ids are rounded on display/entry (but they're correctly handled by sqlite3 in Python and sqlite3 at the cmd line): https://code.google.com/p/sqlite-manager/issues/detail?can=2&start=0&num=100&q=&colspec=ID%20Type%20Status%20Priority%20Milestone%20Owner%20Summary&groupby=&sort=&id=3

License
=======

MIT

Copyright (c) 2013 Ian Ozsvald

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentati
on files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom 
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Sof
tware.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WAR
RANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYR
IGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, A
RISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2013 Ian Ozsvald

