import pandas as pd
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import torch
# import torch_xla.core.xla_model as xm
from collections import Counter
import tensorflow as tf
from unidecode import unidecode
import string
import time
import tensorflow_datasets as tfds

# def move_to_tpu(sample):

#     import torch_xla.core.xla_model as xm

#     device = xm.xla_device()

#     def _move_to_tpu(tensor):
#         return tensor.to(device)

#     return apply_to_sample(_move_to_tpu, sample)

# def get_tpu_device():
#     return xm.xla_device()

# def tpu_data_loader(itr):
#     import torch_xla.core.xla_model as xm
#     import torch_xla.distributed.parallel_loader as pl

#     from fairseq.data import iterators

#     xm.rendezvous("tpu_data_loader")  # wait for all workers
#     xm.mark_step()
#     device = xm.xla_device()
#     return iterators.CountingIterator(
#         pl.ParallelLoader(itr, [device]).per_device_loader(device),
#         start=getattr(itr, "n", 0),
#         total=len(itr),
#     )

# device = xm.xla_device()
# torch.arange(0, 100, device=device)
#from simpletransformers.classification import ClassificationModel

from transformers import AutoTokenizer, AutoModelForMaskedLM,TFBertForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("../bert-base-portuguese-cased")

# model = AutoModelForMaskedLM.from_pretrained("../bert-base-portuguese-cased")

# try:
#     # tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
#     # tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
#     # tf.distribute.TPUStrategy(cluster_resolver)
# except ValueError: # If TPU not found
#     tpu = None

# if tpu:
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
#     print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
# else:
#     strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU
#     print('Running on CPU instead')
# print("Number of accelerators: ", strategy.num_replicas_in_sync)

#try:
 #   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()#'grpc://10.164.0.30')
#    print('Running on TPU ', tpu.cluster_spec().as_dict())  
  #  tf.config.experimental_connect_to_cluster(tpu)
 #   tf.tpu.experimental.initialize_tpu_system(tpu)
 #   strategy = tf.distribute.experimental.TPUStrategy(tpu)
  #  print("Number of accelerators: ", strategy.num_replicas_in_sync)
#except ValueError:
 #   strategy = tf.distribute.get_strategy() # for CPU and single GPU
  #  print('Number of replicas:', strategy.num_replicas_in_sync)

model_tf = TFBertForSequenceClassification.from_pretrained('../bert-base-portuguese-cased')

df_told_full = pd.read_csv('../ToLD-BR.csv')

y = []
for i in range(len(df_told_full)):
      if int(df_told_full.iloc[i,1:].sum())>1:
        y.append(1)
      else:
        y.append(0)
print(np.unique(y))
df_told_full['bin_class'] = y

df_sub_a = df_told_full[df_told_full['bin_class']==0].iloc[0:50]
df_sub_b = df_told_full[df_told_full['bin_class']==1].iloc[0:50]
df_balanced_told = pd.concat([df_sub_a,df_sub_b])

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = unidecode(text)
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    #words = re.findall(r'\b\w+\b', text)

    return text

# Apply the preprocessing function to the DataFrame
df_balanced_told['text'] = df_balanced_told['text'].apply(preprocess_text)
dataset = tf.data.Dataset.from_tensor_slices((df_balanced_told['text'],df_balanced_told['bin_class']))
size = dataset.cardinality().numpy()
train_size = int(1*size)
# test_size = int(size-train_size)
ds_train = dataset.take(train_size)
# ds_test = dataset.skip(train_size)

max_length = 512
batch_size = 256

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )
    
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label
    
def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

# train dataset

start=time.time()
ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)
print("Done with Training Dataset",time.time()-start)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    refer : https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
    
lr = 1e-1

def scheduler(epoch, lr):
  if epoch < 15:
    return lr
  else:
    return lr * 0.5#tf.math.exp(-0.1)
# input_ids = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'input_ids')
# attention_mask = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'attention_mask')
# token_type_ids = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'token_type_ids')

# output = model_tf([input_ids,attention_mask,token_type_ids])
# output = output[0]
# output = tf.keras.layers.Dense(128,activation = gelu)(output)
# output = tf.keras.layers.Dropout(0.1)(output)
# output = tf.keras.layers.Dense(256,activation = gelu)(output)
# output = tf.keras.layers.Dropout(0.2)(output)
# output = tf.keras.layers.Dense(512,activation = gelu)(output)
# output = tf.keras.layers.Dropout(0.2)(output)
# output = tf.keras.layers.Dense(1,activation = 'softmax')(output)

# model = tf.keras.models.Model(inputs = [input_ids,attention_mask,token_type_ids],outputs = output)
# # choosing Adam optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-07)
# optimizer2 = tf.keras.optimizers.AdamW(learning_rate=lr,weight_decay=0.004,epsilon=1e-07)
# # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
# loss = tf.keras.losses.BinaryFocalCrossentropy()
# metric = tf.keras.metrics.BinaryAccuracy('acc')
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=300)
# csv_log = tf.keras.callbacks.CSVLogger('training.log')
# #callbacks=[early_stopping]
# callbacks_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
# checkpoint_filepath = 'modelo_bert_current.h5'
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     monitor='val_loss',
#     save_best_only=False,
#     save_weights_only=False,
#     mode='min',
#     save_freq='epoch',
#     period=3
# )
# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# bert_history = model.fit(ds_train_encoded, epochs=15, validation_data=ds_test_encoded, callbacks=[csv_log,model_checkpoint_callback,early_stopping])
# model.save('modelo_bert_2000_adam_2voters.h5')
# model.compile(optimizer=optimizer2, loss=loss, metrics=[metric])
# bert_history = model.fit(ds_train_encoded, epochs=15, validation_data=ds_test_encoded, callbacks=[csv_log,model_checkpoint_callback,early_stopping])
# model.save('modelo_bert_2000_adamw_2voters.h5')
model = tf.keras.models.load_model("modelo_bert_2000_adamw_2voters.h5", custom_objects={"TFBertForSequenceClassification": TFBertForSequenceClassification})
y_pred = model.predict(ds_train_encoded)
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
plt.savefig("confusionmatrix_told_bert.png", format='png')
def hate_dataset():
    df_hate = pd.read_csv('../gportuguese_hate_speech_binary_classification.csv')
    print("df_hate acquired")
    counts = Counter(df_hate['hatespeech_comb'])
    print(f'full hate: {counts}')
    return df_hate
df_hate = hate_dataset
df_sub_hate = df_hate.iloc[0:100,:]
counts = Counter(df_sub_hate['hatespeech_comb'])
print(f"sub: {counts}")

dataset = tf.data.Dataset.from_tensor_slices((df_sub_hate['text'],df_sub_hate['hatespeech_comb']))
size = dataset.cardinality().numpy()
train_size = int(1*size)
# test_size = int(size-train_size)
ds_train = dataset.take(train_size)
# ds_test = dataset.skip(train_size)
start=time.time()
ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)
print("Done with Training Dataset",time.time()-start)


y_pred = clf.predict(ds_train_encoded)


print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#plt.figure(figsize = (10,7))
ax =sns.heatmap(conf_matrix, annot=True)

# save the plot as PDF file
plt.savefig("confusionmatrix_hate_LSTM_rs.png", format='png')
