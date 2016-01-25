
# coding: utf-8

# In[1]:

from sys import platform as _platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = "./data/"


# In[4]:

#frequencies=pd.read_csv(path+"data_frequences.csv").drop("Unnamed: 0", axis=1)
frequencies=pd.read_csv(path+"fft_eeg.csv")
frequencies_acc =pd.read_csv(path+"fft_acc.csv")

stats=pd.read_csv(path+"data_stat_feats.csv").drop("Unnamed: 0", axis=1)
labels=pd.read_csv(path+"challenge_output_data_training_file_sleep_stages_classification.csv", sep=";")


# In[22]:

def select_freq_names(low, high, X_columns, prefix = ''):
    return [name for name in X_columns 
            if len(name.split('q'))==2 
            and name.split('freq')[0] == prefix
            and low<=float(name.split('freq')[1]) 
            and high>= float(name.split('freq')[1])]
def group_frequencies(name, low, high, frequencies, prefix = ''):
    frequencies[name]=(1./(high-low) * (frequencies[select_freq_names(low,high,frequencies.columns,prefix)])).sum(axis=1)


# In[23]:

new_feat = ["delta", 'theta', 'alpha1','alpha2', 'beta']
#frequencies["delta"]=frequencies[select_freq_names(0,3,frequencies.columns)].sum(axis=1)
#frequencies["delta"]=frequencies[select_freq_names(0,3.99,frequencies.columns)].sum(axis=1)
#frequencies["theta"]=frequencies[select_freq_names(4,7.5,frequencies.columns)].sum(axis=1)
#frequencies["alpha"]=frequencies[select_freq_names(7.5,13.99,frequencies.columns)].sum(axis=1)
#frequencies["beta"]=frequencies[select_freq_names(14,50,frequencies.columns)].sum(axis=1)

def make_new_feats(frequencies):
    group_frequencies("delta", 0.8, 3.99, frequencies)
    group_frequencies("theta", 4, 7.499, frequencies)
    group_frequencies("alpha1", 7.5, 9.5, frequencies)
    group_frequencies("alpha2", 9.5, 13.99, frequencies)
    group_frequencies("beta", 14, 50, frequencies)
make_new_feats(frequencies)



# In[25]:

def regroup_acc_freq (frequencies_acc):
    for prefix in ['ACC_X.','ACC_Y.','ACC_Z.']:
        group_frequencies(prefix+"smaller_one",0.01,1, frequencies_acc,prefix)
        group_frequencies(prefix+"one_to_two",1.01,2, frequencies_acc,prefix)
        group_frequencies(prefix+"two_to_three",2.01,3, frequencies_acc,prefix)
        group_frequencies(prefix+"three_to_four",3.01,4, frequencies_acc,prefix)
        group_frequencies(prefix+"more_four",4,10, frequencies_acc,prefix)
regroup_acc_freq (frequencies_acc)

prefixes = ['ACC_X.','ACC_Y.','ACC_Z.']
frequencies_acc = frequencies_acc[[prefix+ x for x in["smaller_one","one_to_two","two_to_three",'more_four']for prefix in prefixes]]


# In[26]:

train = pd.concat([frequencies[new_feat], stats, frequencies_acc], axis=1)





# In[28]:

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X = train.drop(["skew_ACC_X","skew_ACC_Y", "skew_ACC_Z"], axis=1)
scaler = StandardScaler().fit(X)
X_sc = scaler.transform(X)
X_columns = train.columns
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, labels["TARGET"], test_size=0.2, random_state=0)


from sknn.mlp import Layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sknn.mlp import Classifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score
nn =  nn = Classifier(
            layers=[
                Layer("Rectifier", units=8000),
                Layer("Softmax")],
            learning_rule = 'nesterov', learning_rate=0.04, batch_size = 200,
            dropout_rate = 0.,
            n_iter=20,
            verbose = 1, 
            valid_size = 0.1, 
            n_stable = 15,
            debug = False,
            #    regularize = 'L2'
        )

pipeline_nn = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(0.0, 1.0))),
    ('neural network', nn)
    ])

pipeline_nn.fit(X_train, y_train)
predicted_label = pipeline_nn.predict(X_test)

print("GBC - accuracy Score on test_data : ", accuracy_score(y_test, predicted_label))
print("GBC - kappa Score on test_data : ", cohen_kappa_score(y_test, predicted_label))
print("GBC- kappa Score on train data : ", cohen_kappa_score(y_train, pipeline_nn.predict(X_train)))
