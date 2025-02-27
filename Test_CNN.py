
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import lasagne


# In[9]:

path="data/"
data=pd.read_csv(path+"input_train.csv", index_col="ID")
labels=pd.read_csv(path+"challenge_output_data_training_file_sleep_stages_classification.csv", sep=";", index_col="ID")


# In[5]:

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


# In[15]:

X=data.filter(regex="EEG.*")
y=labels


# In[18]:

# Load the dataset
from sklearn.cross_validation import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.10, random_state=42)
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


# In[18]:

from tempfile import mkstemp


# In[19]:

mkstemp(dir="../tmp")


# In[25]:

import os
os.environ["THEANO_FLAGS"] = "blas.ldflags=-llapack -lblas -lgfortran"


# In[26]:

train_fn =theano.function([input_var, target_var], loss, updates=updates)


# In[ ]:

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# In[ ]:
import time
num_epochs=40
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))


# In[ ]:

predicted_values = test_prediction.flatten()
print "test loss:", test_loss


# In[ ]:



