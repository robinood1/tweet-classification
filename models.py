'''
In this file you can find different implementations of CNN or RNN we tried for our project:
'''
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
import sklearn as sk

import numpy as np

###########################################################################################################################################

'''
Helper functions for CNN and RNN
'''

def model_parameters_cnn():
    ''' 
    Parameters for the 2 CNN models:
    num_filters: number of filter in the convolutional layers
    hidden_dims: dimensions of the dense layer
    batch_size: size of a batch for training
    num_epochs: number of epochs for our training
    filter_sizes: size of our filters in the convolutional layers
    '''
    
    num_filters = 512
    hidden_dims = 100
    # Training parameters
    batch_size = 512
    num_epochs = 10
    filter_sizes = [3,8]
    
    return num_filters, hidden_dims, batch_size, num_epochs, filter_sizes

def preparation_RNN(y):
    '''
    Take the y and replace 0 by [0,1] and 1 by [1,0]
    '''
    
    result =[]
    for elem in y:
        if(elem==0):
            result.append(np.array([0,1]))
        else: result.append(np.array([1,0]))
    return np.array(result)

def prepare_labels_train_test(y_train, y_test):
    
    y_train = preparation_RNN(y_train)
    y_test = preparation_RNN(y_test)
    
    return y_train, y_test
###########################################################################################################################################
'''
Logistic regression
'''

def logistic_regression(x_train,y_train):
    '''
    Create a logistic regression model with c = 0.05, tol=0.0001 and 50 iterations and returns it. 
    '''
    model = sk.linear_model.LogisticRegression(C =0.05,tol=0.00001,max_iter=50,solver='newton-cg',multi_class='ovr',n_jobs=-1)
    model.fit(x_train[:-40000],y_train[:-40000])
    return model

###########################################################################################################################################
'''
CNN using parallel convolution blocks
'''

def cnn_parallel_init(num_filters, hidden_dims, batch_size, num_epochs, filter_sizes, embedding_dim, sequence_length,voc):
    '''
    Create the CNN model using parallel convolutional blocks.
    It has two identical blocks of convolutional layers, batchnormalization layers and maxpooling layers.
    Then it has a flatten layer and two dense layers.
    '''
    
    # Build model
    input_shape = (sequence_length,)

    model_input = Input(shape=input_shape)
    z = Embedding(len(voc), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

    z = Dropout(0.5)(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)

        BatchNormalization()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Dropout(0.1)(conv)


        conv = Convolution1D(filters=int(num_filters/2),
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(conv)
        conv = MaxPooling1D(pool_size=2)(conv)  
        BatchNormalization()(conv)
        conv = Dropout(0.1)(conv)

        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(0.5)(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer= Adam(lr=0.01), metrics=['accuracy'])
    
    return model

###########################################################################################################################################

'''
Simple Sequential CNN
'''

def cnn_sequential_init(num_filters, hidden_dims, batch_size, num_epochs, filter_sizes, embedding_dim, sequence_length,voc):
    '''
    Create the sequential CNN model. 
    With one embedding layer, two convolutional layers followed by a batchnormalization layer and a maxpooling layer.
    This is repeated a second time with a smaller number of filters.
    It has then a flatten layer and three dense layer.
    '''
    
    model = Sequential()

    model.add(Embedding(len(voc), output_dim=embedding_dim, input_length=sequence_length, name="embedding"))

    model.add(Conv1D(num_filters* 2, 8, activation='relu', input_shape=(sequence_length, embedding_dim)))
    model.add(Conv1D(num_filters * 2, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(num_filters, 8, activation='relu'))
    model.add(Conv1D(num_filters, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(1))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(int(hidden_dims/2),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    
    return model

###########################################################################################################################################

'''
RNN model.
'''

def rnn_init(embedding_dim, sequence_length,voc):
    '''
    Create the RNN model.
    With one embedding layer, an LSTM layer of size 350 and a dense layer.
    '''
    
    model = Sequential()
    model.add(Embedding(len(voc), output_dim=embedding_dim, input_length=sequence_length, name="embedding"))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(250, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    
    return model

###########################################################################################################################################

'''
Combined CNN and RNN
'''

def cnn_rnn_init(embedding_dim, sequence_length):
    '''
    Create the combined CNN and RNN model0
    With one embedding layer, followed by a convolutional layer, a LSTM layer and finally a dense layer.
    '''
    model = Sequential()

    model.add(Embedding(len(voc), output_dim=embedding_dim, input_length=sequence_length, name="embedding"))

    model.add(Dropout(0.25))
    model.add(Conv1D(64,
                     8,
                     activation='relu',
                     strides=1))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(250, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    
    return model