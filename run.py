'''
All the imports that you need to run the code
'''

import numpy as np
import random
from scipy.sparse import *
import csv
from implementations import *

# specific imports for our differents CNN and RNN.
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Conv2D, LSTM, Conv1D, SpatialDropout1D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import backend as K
from keras import callbacks
from keras.layers import BatchNormalization
from keras.metrics import binary_accuracy
import tensorflow as tf

from models import *
###########################################################################################################################################
'''
open the needed data. And prepare it for the computations.
'''

# open processed_tweets.txt
with open('processed_tweets.txt', 'r',encoding='utf8') as f:
    processed_tweets = f.readlines()
    
# remove the \n and split each tweet into a list of words
processed_tweets = remove_n(processed_tweets)
length_tweets = int(len(processed_tweets)/2)
x_ = [s.split(" ") for s in processed_tweets]

'''
Matrix Factorization.
'''

#load the cooc matrix
print("loading cooccurrence matrix")
with open('cooc.pkl', 'rb') as f:
    cooc = pickle.load(f)
print("{} nonzero entries".format(cooc.nnz))

#initialize every parameters for the SGD
print("initializing embeddings")
embedding_dim = 300
np.random.seed(seed=15)
tf.set_random_seed(15)
xs = np.random.normal(size=(cooc.shape[0],embedding_dim))
ys = np.random.normal(size=(cooc.shape[1],embedding_dim))
epochs = 20
lambda_ys = 0.01
lambda_xs = 0.01
gamma = 0.000001

# Matrix Factorization using SGD
print("Factorization of the cooc matrix. It can be quite long.")
for epoch in range(epochs):
    print("epoch {}".format(epoch))
    # decrease step size
    gamma /= 1.2
    
    for ix, jy, n in zip(cooc.row,cooc.col,cooc.data):
        logn = np.log(n)
        x, y = xs[ix,:], ys[jy,:]
        error = logn - np.dot(x,y)
        xs[ix,:] += gamma * (error * y - lambda_xs * x)
        ys[jy,:] += gamma * (error * x - lambda_ys * y)
    rmse = compute_error(logn, xs, ys, cooc.row,cooc.col,xs,ys)
    print("iter: {}, RMSE on training set: {}.".format(epoch, rmse))

print("Matrix factorization is over.")

# recover the embeddings and save them to disk
embeddings = xs
np.save('embeddings', xs)

###########################################################################################################################################
'''
Prepare the data for the CNN/RNN.
'''

# open the vocab created previously.
with open('vocab_cut.txt', 'r',encoding='utf8') as f:
    vocab = f.readlines()

# create the labels you are gonna pass to the model.
positive_labels = [1 for _ in range(length_tweets)]
negative_labels = [0 for _ in range(length_tweets)]
y = np.concatenate([positive_labels, negative_labels], 0)

# remove the \n from the vocab.
vocabulary = remove_n(vocab)

# add a padding word and vector to both embedding matrix and vocabulary.
embeddings = np.insert(embeddings,0,[0 for i in range(300)],axis=0)
# create a MAP where each word is the key corresponding to an index starting at 1.
voc = {x: i + 1  for i, x in enumerate(vocabulary)}
voc['<PAD>'] = 0

# replace the words in the tweet list by the corresponding index in the vocabulary MAP.
print("replacing words by index.")
x_final = replace_word_by_key(x_,voc)
sequence_length = max(len(x) for x in x_final)
print("the longuest tweet has:")
print(sequence_length, "words")

# pad the tweets.
x_final = padding(x_final,sequence_length)
x_final = np.array(x_final)

###########################################################################################################################################
'''
CNN or RNN implementations. Here you have access only to our best models.
You can access all the models we tried in the file model.txt
'''

# to run on GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# verify that you are using the GPU to train the CNN\RNN.
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

np.random.seed(seed=15)

# split the data into training data and testing data.
y_ = y
shuffle_indices = np.random.permutation(np.arange(len(y_)))
x = x_final[shuffle_indices]
y_ = y_[shuffle_indices]
train_len = int(len(x_) * 0.8)
x_train = x[:train_len]
y_train = y_[:train_len]
x_test = x[train_len:]
y_test = y_[train_len:]

# transform the labels from 0 and 1 to a bit representation : [0,1] and [1,0].
y_train, y_test= prepare_labels_train_test(y_train, y_test)

# compile the rnn model.
print( "create the model:")
model = rnn_init(embedding_dim, sequence_length,voc)
print(model.summary())

# setup embedding layer of the model.
weights = np.array([v for v in embeddings])
print("Initializing embedding layer with weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

# Train the model and create checkpoints to continues the training later on.
checkpoint = callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit(x_train, y_train, batch_size=512, epochs=10,
          validation_data=(x_test, y_test), verbose=1,callbacks=callbacks_list)

###########################################################################################################################################
'''
Create the submission.
'''

# open the test file.
with open('test_data.txt', 'r',encoding='utf8') as f:
    test = f.readlines()
    
# create the labels.
test_id = np.arange(1,len(test)+1)

# process the test data as previously but with a new function that removes the labels.
print("process test data.")
proc_test = process_tweets_test(list(test))

# Split by words
x_test = proc_test
x_test = [s.split(" ") for s in x_test]

# replace the words in the tweet list by the corresponding index in the vocabulary MAP.
print("replacing words in test data.")
x_final_test = replace_word_by_key(x_test,voc)

# pad the tweets.
x_final_test = padding(x_final_test,sequence_length)
x_final_test = np.array(x_final_test)

# predicting the labels for the test data
print('predicting the labels:')
final_labels = (model.predict(x_final_test))
print("done.")

# from probabilities, creating the correct labels for the test data
final_classes = np.zeros(len(test_id))
for i in range(final_labels.shape[0]):
    proba = final_labels[i][0]
    if proba >= 1/2:
        final_classes[i] = 1
    else:
        final_classes[i] = -1
        
# create the csv submission
create_csv_submission(test_id,final_classes,'test-submission_.csv')