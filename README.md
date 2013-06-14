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

    $ pip install requirements.txt  # base libs
    $ pip install requirements_matplotlib.txt  # get around MEP11 with second requirements file

Note that generally I'm using Python 3 __future__ imports, the code isn't tested with Python 3 but the porting should be straight-forward. sqlite only wants byte/strings for key indexing (not unicode strings).

Tests
-----

    $ nosetests  # the project defaults to 'testing' if DISAMBIGUATOR_CONFIG isn't set
    $ nosetests -s  # runs without capturing stdout, useful if you're using `import pdb; pdb.set_trace()` for debugging
    $ nosetests --with-coverage --cover-html  # with an HTML coverage report to cover/index.html

TO FIX
------

  * the OpenCalais NER code has 1 HARDCODED ENTRY for 'apple' at present, this obviously needs to be extracted as a parameter.
  * tweet_annotator.py has some hardcoded apple entries


Creating a gold standard
------------------------

    $ u'/home/ian/workspace/virtualenvs/tweet_disambiguation_project/prototype1/src'
    $ #export DISAMBIGUATOR_CONFIG=production  # might be useful if not using ipython
    $ DISAMBIGUATOR_CONFIG=production ipython
    $ %run tweet_annotator.py ../../apple10.json apple
    # or
    $ DISAMBIGUATOR_CONFIG=production python tweet_annotator.py ../../apple10.json apple


Annotating the tweets using OpenCalais
--------------------------------------

OpenCalais have a strong named entity recognition API offering, we can use it to annotate tweets to see how it does.

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

Results are written to a hardcoded table name (see learn1.py)

******* NOTE use SQLiteManager to copy scikit_validation_app to learn1_validation_apple

    $ DISAMBIGUATOR_CONFIG=production python learn1.py scikit_testtrain_apple 

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

  * Take my list of tweet-ids, fetch them from Twitter and insert into SQLite using sql_convenience, add the classification that I'll also provide (note - ask me for this list as it isn't in the repo yet)
  * Extract a validation set of 100 in- and 100 out-of-class tweets, send to OpenCalais, verify that my OpenCalais parsing code is reading their responses correctly - did I miss any answers? Adding more fixtures to test_opencalais_ner.py would be a good start
  * Replicate the OpenCalais code for DBPediaSpotlight so we have a second public NER engine to compare against
  * Extend ner_api_caller.py and build a local simple "capitalised brand detector" (e.g. it only looks for the exact term "Apple" in the tweet) - it'll do a similar job to the OpenCalais and DBPedia API calls but without an API call as the code will be local (and really simple) - this would be a very sensible baseline tool

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

