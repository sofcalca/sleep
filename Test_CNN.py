
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import lasagne


# In[7]:

path="data/"
data=pd.read_csv(path+"input_train.csv")
labels=pd.read_csv(path+"challenge_output_data_training_file_sleep_stages_classification.csv", sep=";")


# In[8]:

def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 3750, 1),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(25, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(5, 1))
    #network = lasagne.layers.Conv2DLayer(
    #    network, num_filters=32, filter_size=(10, 1),
    #    nonlinearity=lasagne.nonlinearities.rectify,
    #    W=lasagne.init.GlorotUniform())
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(5, 1))
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=5,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


# In[9]:

# Load the dataset
from sklearn.cross_validation import train_test_split
X_train, y_train, X_test, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
# Create neural network model
network = build_cnn(input_var)


# In[10]:

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.multiclass_hinge_loss(prediction, target_var)
loss = loss.mean()


# In[12]:

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)


# In[13]:

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()


# In[14]:

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)


# In[ ]:

train_fn = theano.function([input_var, target_var], loss, updates=updates)


# In[ ]:



