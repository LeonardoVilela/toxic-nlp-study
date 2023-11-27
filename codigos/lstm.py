#!pip install simpletransformers --quiet

# pip install --upgrade pip setuptools wheel
# !pip install transformers

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
from sklearn.svm import SVC 
# import torch_xla.core.xla_model as xm
from sklearnex import patch_sklearn
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import pickle
import joblib
import tensorflow as tf

patch_sklearn()

# device = xm.xla_device()
# torch.arange(0, 100, device=device)
#from simpletransformers.classification import ClassificationModel

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("../bert-base-portuguese-cased")

model = AutoModelForMaskedLM.from_pretrained("../bert-base-portuguese-cased")

class BertTokenizer(object):
    def __init__(self, text=[]):
        self.text = text
        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer#.to(device)
        self.model = model#.to(device)
    def get(self):
        df = pd.DataFrame(data={"text":self.text})
        tokenized = df["text"].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)#.to(device)#,device = device)
        attention_mask = torch.tensor(attention_mask)#.to(device)#,device = devic)
        with torch.no_grad(): last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features


#print("Predictions:", predictions)
#print("Outputs:", outputs)

#rodar no dataset e ver a precisao media disso
def told_dataset(tamanho):
    df_told = pd.read_csv('../ToLD-BR.csv')
    print("told acquired")
    
    y =[]
    for i in range(len(df_told)):
      if int(df_told.iloc[i,1:].sum())>1:
        y.append(1)
      else:
        y.append(0)
    print(np.unique(y))
    df_told['bin_class'] = y
    # df_told = df_told[0:5000]
    print("told bin acquired")
    df_sub_a = df_told[df_told['bin_class']==0].iloc[0:tamanho]
    df_sub_b = df_told[df_told['bin_class']==1].iloc[0:tamanho]
    df_full = pd.concat([df_sub_a,df_sub_b])#.reset_index(inplace=True)
    df_full.reset_index(inplace=True)
    counts = Counter(df_full['bin_class'])
    print(counts)
    return df_told,df_full
    
df_told,df_full = told_dataset(3816)

def train_tokens(df_full):
    # _instance = BertTokenizer(text=list(df_full['text']))
    # X_train = _instance.get()
    X_train = np.loadtxt('X_train.txt')
    print(X_train.shape)
    print("X_train acquired")
    y_train = df_full['bin_class']#.iloc[0:250]
    return X_train,y_train

X_train,y_train = train_tokens(df_full)
print(type(X_train))
# np.savetxt('X_train.txt',X_train)
def teste_tokens(df_told,tamanho):
    inicio = 3816
    final = inicio+tamanho
    df_sub_a = df_told[df_told['bin_class']==0].iloc[inicio:final]
    df_sub_b = df_told[df_told['bin_class']==1].iloc[inicio:final]#.iloc[8000:]
    df_full = pd.concat([df_sub_a,df_sub_b])#.reset_index(inplace=True)
    df_full.reset_index(inplace=True)
    y_test = df_full['bin_class']#.iloc[250:375]
    
    # _instance =BertTokenizer(text=list(df_full['text']))
    # X_test = _instance.get()
    X_test = np.loadtxt('X_test.txt')
    print(X_test.shape)
    print("X_test acquired")
    return X_test,y_test
    
X_test,y_test=teste_tokens(df_told,999)
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
horizon = 3
clf = Sequential()
clf.add(LSTM(30, activation='tanh', input_shape=(X_test.shape[1],1)))
clf.add(Dense(units=64, activation='tanh'))
clf.add(Dense(units=128, activation='tanh'))
clf.add(Dense(1,activation = 'softmax'))
clf.compile(loss='BinaryFocalCrossentropy', optimizer='adam',metrics = [tf.keras.metrics.F1Score(name = 'f1')])
train = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))
train = train.batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test.astype(np.float32)))
test_dataset = test_dataset.batch(64)
clf.fit(train, epochs=50, batch_size=64,validation_data=test_dataset,callbacks=tf.keras.callbacks.EarlyStopping(monitor='f1', patience=50))

# save model
# clf = pickle.load(open('rf_model_nors.pickle', 'rb'))
clf.save('lstm.h5')
y_pred = clf.predict(X_test)
print(y_pred.shape)
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix, annot=True)

ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_told_LSTM_rs.png", format='png')

# classifier = xgboost.XGBClassifier()
# random_grid = {
#  "learning_rate" : [x for x in np.linspace(start = 1e-6, stop = 1e-2, num = 1000)],
#  "max_depth" : [int(x) for x in np.linspace(10, 5000, num = 1000)],
#  "min_child_weight" : [int(x) for x in np.linspace(1, 500, num = 500)],
#  "gamma": [x for x in np.linspace(start = 2e-3, stop = 7e-1, num = 100)],
#  "colsample_bytree" : [x for x in np.linspace(start = 2e-3, stop = 7e-1, num = 100)]
# }
# def xgboost_rs(random_grid,classifier):
#     xg_random = RandomizedSearchCV(estimator = classifier,scoring = ['accuracy','f1'],param_distributions = random_grid, n_iter = n_iter, cv = 5, verbose=10, random_state=42, n_jobs = -1,refit='f1')
#     # Fit the random search model
#     # xg_random.set_params(**fit_params)
#     xg_random.fit(X_train, y_train)#,**fit_params)
#     #clf.fit(X_train, y_train)
#     print("xg model acquired")
#     xg_clf = xg_random.best_estimator_
#     return xg_clf
# xg_clf = xgboost_rs(random_grid,classifier)

# # save model
# joblib.dump(xg_clf, 'xg_model_joblib.pkl')
# pickle.dump(xg_clf, open("xg_model.sav", "wb"))
# pickle.dump(xg_clf, open("xg_model.pickle", "wb"))
# y_pred = xg_clf.predict(X_test)
# print('Precision: %.3f' % precision_score(y_test, y_pred))
# print('Recall: %.3f' % recall_score(y_test, y_pred))
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
# print('F1 Score: %.3f' % f1_score(y_test, y_pred))

# conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
# plt.figure(figsize = (10,7))
# sns.heatmap(conf_matrix, annot=True)

# ax =sns.heatmap(conf_matrix, annot=True)

# # save the plot as PDF file
# plt.savefig("confusionmatrix_told_xg_rs.png", format='png')


def hate_dataset():
    df_hate = pd.read_csv('../gportuguese_hate_speech_binary_classification.csv')
    print("df_hate acquired")
    counts = Counter(df_hate['hatespeech_comb'])
    print(f'full hate: {counts}')
    return df_hate
df_hate = hate_dataset
df_sub_hate = df_hate.iloc[0:1000,:]
counts = Counter(df_sub_hate['hatespeech_comb'])
print(f"sub: {counts}")

X_test_hate = list(df_hate['text'])
_instance =BertTokenizer(text=X_test_hate)
X_test_hate = _instance.get()
y_test = list(df_hate['hatespeech_comb'])

y_pred = clf.predict(X_test_hate)


print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#plt.figure(figsize = (10,7))
ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_hate_LSTM_rs.png", format='png')

X_test_hate_sub = list(df_sub_hate['text'])
_instance =BertTokenizer(text=X_test_hate_sub)
X_test_hate_sub = _instance.get()
y_test_sub = list(df_sub_hate['hatespeech_comb'])

y_pred = clf.predict(X_test_hate_sub)

print('Precision: %.3f' % precision_score(y_test_sub, y_pred))
print('Recall: %.3f' % recall_score(y_test_sub, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test_sub, y_pred))
print('F1 Score: %.3f' % f1_score(y_test_sub, y_pred))

conf_matrix = confusion_matrix(y_true=y_test_sub, y_pred=y_pred)
#plt.figure(figsize = (10,7))
ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_hate_sub1000_LSTM_rs.png", format='png')

# y_pred = xg_clf.predict(X_test_hate)


# print('Precision: %.3f' % precision_score(y_test, y_pred))
# print('Recall: %.3f' % recall_score(y_test, y_pred))
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
# print('F1 Score: %.3f' % f1_score(y_test, y_pred))

# conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
# #plt.figure(figsize = (10,7))
# ax =sns.heatmap(conf_matrix, annot=True)

# # save the plot as PDF file
# plt.savefig("confusionmatrix_hate_xg_rs.png", format='png')

y_pred = clf.predict(X_test_hate_sub)

print('Precision: %.3f' % precision_score(y_test_sub, y_pred))
print('Recall: %.3f' % recall_score(y_test_sub, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test_sub, y_pred))
print('F1 Score: %.3f' % f1_score(y_test_sub, y_pred))

conf_matrix = confusion_matrix(y_true=y_test_sub, y_pred=y_pred)
#plt.figure(figsize = (10,7))
ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_hate_sub1000_LSTM_rs.png", format='png')
