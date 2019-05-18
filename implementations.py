''' 
Here you can find all the helper functions we used in the project.
'''

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
###########################################################################################################################################
'''
Functions used for data preprocessing.
'''

def process_tweet(tweet):
    '''
    taking a tweet, split it in word, remove the \n at the end and then remove all digits, float and punctuation from it. It also               transform the occurences of hahahaha , hahaha into the only haha.
    Return the processed tweet.
    '''
    tokens = tweet.split(" ")
    processed_tweet = ""
    for w in tokens:
        if(len(w) > 1):
            w = w.replace('\n', '')
            try:
                float(w)
            except ValueError:
                if ((w[0] == '#') | (w[0] not in string.punctuation)) & (not w.isdigit()):
                    if(not (w[0].isdigit() & w[len(w)-1].isdigit())):
                        if w.find("haha") != -1 :
                            w = "haha"
                        processed_tweet += w + " "
    return processed_tweet[:-1]


def create_sets():
    '''
    Create the differents set of words that have a similar meaning in the different tweets.
    Return sets
    '''
    set_laugh = ['lol','xd','laugh','haha','funny', 'hehe',"fun", "lmfao", "laughing",'lmao',';p',':p','jk','kidding']

    set_good = ['follow','please', 'beautiful', 'great', 'glad', 'enjoyed','nice','better','sweet','amazing','cute','awesome'
           ,'beautiful','friend','follow','pretty','yes']

    set_happy = ['smile','excited', ':)','joy', 'cool', '#happy','grin','lit', ':D', '(:', 'yolo', 
             '420', '4:20', '4|20', 'ftw','thankyou','thanks','thank','thx','bday','birthday','hey','pleasure']

    set_love = ['loove', 'looove','xx', 'xoxo','<3', 'friend', 'friendship',
            ":')", 'awww','aww', 'luv', '#love', 'xxx','loving']

    set_sad = ['cry', 'crying', ';(',':(', ":'(", "sucks",'lonely','broke', 'death', 'wish'
           'rip', ':x','depressing','#depressing', 'fml','ugh', 'suicide','sorry', '#sad', 'miss']

    set_angry = ['fucker' , 'bitch', 'fck', 'pissed', 'hate','mofo', 'anger', 'kill', 'ugly']

    set_bad = ['omg','omfg','sucks','suck','lame', 'pathetic','meh', 'ugh','dentist', '#dentist', 'drama','bored',
           'wrong', 'sick','white','black','frame','pack','complete','edition','wood','poster','book','dvd','series','cd'
          'audio']

    set_neutral = ['fuck','weird','drink','dress','friends','concert','get','know','please','one','see','back','go','got'
              ,'always','like']
    
    return set_laugh, set_good, set_happy, set_love, set_sad, set_angry, set_bad, set_neutral


def replace_words(tweet):
    '''
    Replace all the occurences of a word contained in one of the given sets by the representant of this set.
    The set neutral exists if you want to speed up the computations a lot, but you will loose some accuracy afterwards.
    Return the tweet with replaced words.
    '''
    tokens = tweet.split(" ")
    set_laugh, set_good, set_happy, set_love, set_sad, set_angry, set_bad, set_neutral = create_sets()
    replace_tweet = ""
    for w in tokens:
        if(len(w) > 1):
            w = w.replace('\n', '')
                
            if w.lower() in set_laugh:
                replace_tweet+= 'lol '
            elif w.lower() in set_happy:
                replace_tweet+= 'happy '
            elif w.lower() in set_love:
                replace_tweet+= 'love '
            elif w.lower() in set_sad:
                replace_tweet+= 'sad '
            elif w.lower() in set_angry:
                replace_tweet+= 'angry '
            elif w.lower() in set_good:
                replace_tweet+= 'good '
            elif w.lower() in set_bad:
                replace_tweet+= 'bad '
            else:
                replace_tweet+= w + " "
    return replace_tweet[:-1]


def process_tweets(tweets):
    '''
    Taking a list of tweets, processed each tweet using the functions replace_words process_tweet.
    Return a list of processed tweets.
    '''
    i=0
    result = []
    first = ""
    for t in tweets:
        first = replace_words(t)
        result.append(process_tweet(first))
        i+=1
    return result


def remove_n(data):
    '''
    Remove the occurences of the \n at the end of a line when opening a file.
    This function is used when loading already processed data so that we don't have to do the all processing again.
    '''
    for i in range(len(data)):
        data[i] = data[i].replace('\n', '')
    return data

###########################################################################################################################################
'''
Helper function for the matrix factorization.
'''

def compute_error(data, user_features, item_features, cooc_row,cooc_col,xs,ys):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in zip(cooc_row,cooc_col):
        x, y = xs[row, :], ys[col, :]
        mse += (data - np.dot(x, y)) ** 2
    return np.sqrt(1.0 * mse / len(cooc_row))

###########################################################################################################################################
'''
Helper function for the data processing before entering the CNN/RNN models.
'''

def replace_word_by_key(tweets,voc):
    '''
    For each tweet, replace each words by the corresponding index in the vocabulary map.
    For example replace 'the' by the index 1.
    Drop words that are not in the vocabulary map.
    '''
    tweets_final = []
    index = 0
    for tweet in tweets:
        new_tweet = []
        for word in tweet:
            try:
                index = voc[word]
                new_tweet.append(index)
            except:
                error = 1
        tweets_final.append(new_tweet)
    return tweets_final


def padding(tweets,sequence_length):
    '''
    add padding to each tweet so that they all have the same length.
    The tweets here are in numeric forms (one index per word).
    '''
    tweets_final = []
    for tweet in tweets:
        num_padding = sequence_length - len(tweet)
        padding = [0 for i in range(num_padding)]
        new_tweet = tweet
        new_tweet.extend(padding)
        tweets_final.append(new_tweet)
    return tweets_final

###########################################################################################################################################
'''
Helper function for the data processing before submitting.
'''

def process_tweet_test(tweet):
    '''
    taking a tweet, split it in word, remove the \n at the end and then remove all digits, float and punctuation from it. It also               transform the occurences of hahahaha , hahaha into the only haha.
    Remove the index at the start of each tweet.
    Return the processed tweet.
    '''
    tokens = tweet.split(" ")
    processed_tweet = ""
    for w in tokens:
        if(len(w) > 1):
            w = w.replace('\n', '')
            try:
                float(w)
            except ValueError:
                if ((w[0] == '#') | (w[0] not in string.punctuation)) & (not w.isdigit()):
                    if(not (w[0].isdigit() & w[len(w)-1].isdigit())):
                        if w.find("haha") != -1 :
                            w = "haha"
                        processed_tweet += w + " "
    return processed_tweet[:-1]

def process_tweets_test(tweets):
    '''
    Taking a list of tweets, processed each tweet using the functions replace_words process_tweet.
    Return a list of processed tweets.
    '''
    i=0
    result = []
    first = ""
    for t in tweets:
        first = replace_words(t)
        result.append(process_tweet_test(first))
        i+=1
    return result

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
