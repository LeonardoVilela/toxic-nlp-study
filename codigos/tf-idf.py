import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import pickle
import joblib

# device = xm.xla_device()
# torch.arange(0, 100, device=device)
#from simpletransformers.classification import ClassificationModel

from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode

# %%
df_told = pd.read_csv('../ToLD-BR.csv')
df_told

# %%
y =[]
for i in range(len(df_told)):
  if df_told.iloc[i,1:].sum()>0:
    y.append(1)
  else:
    y.append(df_told.iloc[i,1:].sum())
np.unique(y)
df_told['bin_class'] = y
df_told = df_told#[0:5000]

# %%
nlp = spacy.load('pt_core_news_lg')
textos = [unidecode(x) for x in list(df_told['text'])]
textos_tokenized = []
for separado in textos:
  textos_middle = []
  doc = nlp(separado)
  for token in doc:
    if not token.is_stop:
      textos_middle.append(token.text)
  textos_tokenized.append(textos_middle)

stemmer = SnowballStemmer(language='portuguese')
textos_tokenized_clean_stem = []
for i in range(len(textos_tokenized)):
  textos_middle =[]
  for j in textos_tokenized[i]:
    textos_middle.append(stemmer.stem(j))
  textos_tokenized_clean_stem.append(textos_middle)

textos_tokenized_clean_lemma =[]
for i in range(len(textos_tokenized)):
  separado = ' '.join(textos_tokenized[i])
  doc = nlp(separado)
  textos_middle =[]
  for token in doc:
    textos_middle.append(token.lemma_)
  textos_tokenized_clean_lemma.append(textos_middle)
textos_lemma = [' '.join(x) for x in textos_tokenized_clean_lemma]
textos_stemma = [' '.join(x) for x in textos_tokenized_clean_stem]

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
def tokenize_spacy(sentence):
    doc = nlp(sentence)
    return [w.text for w in doc]
vectorizer = TfidfVectorizer(tokenizer=tokenize_spacy, ngram_range=(1, 4),lowercase = True)
matrix = vectorizer.fit_transform(textos_lemma)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrix, y, test_size=0.2, random_state=42)

# %%
n_iter = 5
clf = RandomForestClassifier(max_depth=100, random_state=42,n_estimators=500)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 1000)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt',None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 500, num = 1000)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(1, 100, num = 1000)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(10, 100, num = 1000)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_dict_rf = {'n_estimators': [],
               'max_features': [],
               'max_depth': [],
               'min_samples_split': [],
               'min_samples_leaf': [],
               'bootstrap': [],
                'f1':[],
                 'acc':[]}
early_stop_flag = 0
for i in range(0,n_iter):
    print(f'iteration:{i}, early stopping = {early_stop_flag}')
    if i>=2:
        if random_dict_rf['f1'][i-2]>=random_dict_rf['f1'][i-1]:
            early_stop_flag+=1
    if early_stop_flag >=5:
        break
    n_estimators_ = np.random.choice(n_estimators)
    random_dict_rf['n_estimators'].append(n_estimators_)
    max_features_ = np.random.choice(max_features)
    random_dict_rf['max_features'].append(max_features_)
    max_depth_ = np.random.choice(max_depth)
    random_dict_rf['max_depth'].append(max_depth_)
    min_samples_split_ = np.random.choice(min_samples_split)
    random_dict_rf['min_samples_split'].append(min_samples_split_)
    min_samples_leaf_ = np.random.choice(min_samples_leaf)
    random_dict_rf['min_samples_leaf'].append(min_samples_leaf_)
    bootstrap_ = np.random.choice(bootstrap)
    random_dict_rf['bootstrap'].append(bootstrap_)
    random_grid = {'n_estimators': n_estimators_,
               'max_features': max_features_,
               'max_depth': max_depth_,
               'min_samples_split': min_samples_split_,
               'min_samples_leaf': min_samples_leaf_,
               'bootstrap': bootstrap_}
    clf = RandomForestClassifier(**random_grid)
    clf.fit(X_train, y_train)
    print("RF model acquired")
    y_pred = clf.predict(X_test)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    random_dict_rf['acc'].append(accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred,average = 'micro'))
    random_dict_rf['f1'].append(f1_score(y_test, y_pred,average = 'micro')))
    
best_iter = random_dict_rf['f1'].index(max(random_dict_rf['f1']))
n_estimators = random_dict_rf['n_estimators'][best_iter]
max_features = random_dict_rf['max_features'][best_iter]
max_depth = random_dict_rf['max_depth'][best_iter]
min_samples_split = random_dict_rf['min_samples_split'][best_iter]
min_samples_leaf = random_dict_rf['min_samples_leaf'][best_iter]
bootstrap = random_dict_rf['bootstrap'][best_iter]
random_grid = {'n_estimators': n_estimators_,
             'max_features': max_features_,
             'max_depth': max_depth_,
             'min_samples_split': min_samples_split_,
             'min_samples_leaf': min_samples_leaf_,
             'bootstrap': bootstrap_}
clf = RandomForestClassifier(**random_grid)
clf.fit(X_train, y_train)

filename = "rf_model_tfidf_.pickle"

# save model
# clf = pickle.load(open('rf_model_nors.pickle', 'rb'))
joblib.dump(clf, 'rf_model_rs_tfidf_.pkl')
pickle.dump(clf, open("rf_model_rs_tfidf_.sav", "wb"))
pickle.dump(clf, open("rf_model_rs_tfidf_.pickle", "wb"))
y_pred = clf.predict(X_test)
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix, annot=True)

ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_told_rf_rs_tfidf_.png", format='png')

# %%
classifier = xgboost.XGBClassifier()
random_grid = {
 "learning_rate" : [x for x in np.linspace(start = 1e-6, stop = 1e-2, num = 1000)],
 "max_depth" : [int(x) for x in np.linspace(10, 5000, num = 1000)],
 "min_child_weight" : [int(x) for x in np.linspace(1, 500, num = 500)],
 "gamma": [x for x in np.linspace(start = 2e-3, stop = 7e-1, num = 100)],
 "colsample_bytree" : [x for x in np.linspace(start = 2e-3, stop = 7e-1, num = 100)]
}
def xgboost_rs(random_grid,classifier):
    xg_random = RandomizedSearchCV(estimator = classifier,scoring = ['accuracy','f1'],param_distributions = random_grid, n_iter = n_iter, cv = 5, verbose=10, random_state=42, n_jobs = -1,refit='f1')
    # Fit the random search model
    # xg_random.set_params(**fit_params)
    xg_random.fit(X_train, y_train)#,**fit_params)
    #clf.fit(X_train, y_train)
    print("xg model acquired")
    xg_clf = xg_random.best_estimator_
    return xg_clf
xg_clf = xgboost_rs(random_grid,classifier)

# save model
joblib.dump(xg_clf, 'xg_model_tfidf_joblib.pkl')
pickle.dump(xg_clf, open("xg_model_tfidf_.sav", "wb"))
pickle.dump(xg_clf, open("xg_model_tfidf_.pickle", "wb"))
y_pred = xg_clf.predict(X_test)
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix, annot=True)

ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_told_xg_rs_tfidf_.png", format='png')

# %%



