import numpy as np
import pickle
import string
import random
from scipy.sparse import *
import csv
import scipy as sp
from collections import Counter
import itertools
from itertools import groupby
from implementations import *
############################################################################################################################################

'''
open the data
'''

# read the positive train tweets
with open('train_pos_full.txt', 'r',encoding='utf8') as f:
    pos_tweets = f.readlines()
    
# read the negative train tweets 
with open('train_neg_full.txt', 'r',encoding='utf8') as f:
    neg_tweets = f.readlines()
   
'''
Process the tweets
'''
processed_pos_tweets = process_tweets(list(set(pos_tweets)))
processed_neg_tweets = process_tweets(list(set(neg_tweets)))

# create a list of the full tweets data
processed_tweets = processed_pos_tweets + processed_neg_tweets

# write to processed_pos_tweets_fact.txt
with open('processed_pos_tweets_fact.txt', 'w',encoding='utf8') as f:
    for item in processed_pos_tweets:
        f.write("%s\n" % item)
        
# write to processed_neg_tweets.txt
with open('processed_neg_tweets_fact.txt', 'w',encoding='utf8') as f:
    for item in processed_neg_tweets:
        f.write("%s\n" % item)

# write to processed_tweets.txt
with open('processed_tweets.txt', 'w',encoding='utf8') as f:
    for item in processed_tweets:
        f.write("%s\n" % item)