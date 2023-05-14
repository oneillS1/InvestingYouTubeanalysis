
"""
    Installing and loading the necessary packages for the 'Analysis of Investing
    on YouTube' project
"""

# Installing (to be commented out once code is run once - included to help re-run analysis for others)

# Loading the packages
import spacy
from spacy.lang.en import English
import selenium
import gensim
import lazypredict
import pandas as pd
import numpy as np
import py_youtube

series1 = pd.Series([1,2,3,4], index=['a', 'b', 'c', 'd'])
print(series1)


import nltk

print('This worked')