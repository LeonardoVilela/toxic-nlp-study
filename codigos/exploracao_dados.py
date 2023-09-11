#!pip install simpletransformers --quiet

# pip install --upgrade pip setuptools wheel
# !pip install transformers

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
torch.arange(0, 100, device=device)
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
        input_ids = torch.tensor(padded).to(device)#,device = device)
        attention_mask = torch.tensor(attention_mask).to(device)#,device = devic)
        with torch.no_grad(): last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features

#model = ClassificationModel("distilbert", "/content/drive/MyDrive/TCC/toxic_bert_model",use_cuda=False)

#predictions, outputs = model.predict(["eu estava andando na rua e vi dois homens se beijando, achei nojento demais"])
#predictions, outputs = model.predict(["marcelo é um racista"])
#predictions, outputs = model.predict(["seu deus odeia gays"])#/seu Deus odeia gays
#predictions, outputs = model.predict(["essas travestis são muito massa"])


#print("Predictions:", predictions)
#print("Outputs:", outputs)

#rodar no dataset e ver a precisao media disso
df_told = pd.read_csv('../ToLD-BR.csv')
print("told acquired")

y =[]
for i in range(len(df_told)):
  if df_told.iloc[i,1:].sum()>0:
    y.append(1)
  else:
    y.append(df_told.iloc[i,1:].sum())
df_told['bin_class'] = y
df_told = df_told[0:5000]
print("told bin acquired")
df_sub_a = df_told[df_told['bin_class']==0].iloc[0:250]
df_sub_b = df_told[df_told['bin_class']==1].iloc[0:250]
df_full = pd.concat([df_sub_a,df_sub_b])#.reset_index(inplace=True)
df_full.reset_index(inplace=True)

from collections import Counter

counts = Counter(df_full['bin_class'])
print(counts)

_instance =BertTokenizer(text=list(df_full['text']))
X_train = _instance.get()
print("X_train acquired")
y_train = df_full['bin_class']#.iloc[0:250]

df_sub_a = df_told[df_told['bin_class']==0].iloc[251:350]
df_sub_b = df_told[df_told['bin_class']==1].iloc[251:350]
df_full = pd.concat([df_sub_a,df_sub_b])#.reset_index(inplace=True)
df_full.reset_index(inplace=True)
y_test = df_full['bin_class']#.iloc[250:375]

_instance =BertTokenizer(text=list(df_full['text']))
X_test = _instance.get()
print("X_test acquired")
from sklearn.model_selection import RandomizedSearchCV
clf = RandomForestClassifier(max_depth=100, random_state=42,n_estimators=100)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 100)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt',None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000, num = 100)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(1, 100, num = 100)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(10, 100, num = 100)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=10, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
# clf.fit(X_train, y_train)
print("RF model acquired")
clf = rf_random.best_estimator_
y_pred = clf.predict(X_test)
#y_test = df_full['bin_class'].iloc[250:375]
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#plt.figure(figsize = (10,7))
# sns.heatmap(conf_matrix, annot=True)

ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_told.png", format='png')

# df_told = df_told_full.iloc[0:1000,:]

# X = list(df_told['text'])
# from IPython.display import clear_output
# y_pred_told = []
# for i in range(len(X)):
#   clear_output(wait=True)
#   print(f'{round((i/len(X))*100,2)}%')
#   predictions, outputs = model.predict(X[i])
#   y_pred_told.append(predictions[0])
# y_ = np.array(y_pred_told)
# # np.savetxt("y_pred_told.txt", y_)

# y_pred = y
# y_test = df_told['bin_class']
# print('Precision: %.3f' % precision_score(y_test, y_pred_told))
# print('Recall: %.3f' % recall_score(y_test, y_pred_told))
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_told))
# print('F1 Score: %.3f' % f1_score(y_test, y_pred_told))

# conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_told)
# #plt.figure(figsize = (10,7))
# sns.heatmap(conf_matrix, annot=True)

df_hate = pd.read_csv('../2019-05-28_portuguese_hate_speech_binary_classification.csv')
print("df_hate acquired")

df_sub_hate = df_hate.iloc[0:1000,:]

X_test = list(df_sub_hate['text'])
y_test = list(df_sub_hate['hatespeech_comb'])

# from IPython.display import clear_output
y_pred = []
for i in range(len(X_test)):
  print(f'{round((i/len(X_test))*100,2)}%')
  predictions, outputs = model.predict(X_test[i])
  y_pred.append(predictions[0])

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#plt.figure(figsize = (10,7))
ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_hate.png", format='png')

#divir dataset em treino-teste-validacao e testar o modelo (ver a acuracia) ok
# fazer o teste no TOLD e no outro e ver como o modelo lida ok
#ler os artigos ok
#criar um modelinho nosso aqui (naive-bayes)
#ver e tentar rodar a opção multilabel
#testar com o bertimbau
