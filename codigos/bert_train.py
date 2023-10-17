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

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://10.164.0.30')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

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

df_sub_a = df_told_full[df_told_full['bin_class']==0].iloc[0:4816]
df_sub_b = df_told_full[df_told_full['bin_class']==1].iloc[0:4816]
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
train_size = int(0.85*size)
test_size = int(size-train_size)
ds_train = dataset.take(train_size)
ds_test = dataset.skip(train_size)

max_lenght = 512
batch_size = 64

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
# test dataset
start=time.time()
ds_test_encoded = encode_examples(ds_test).batch(batch_size)
print("Done with Testing Dataset",time.time()-start)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

lr = 2e-2

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * 0.5#tf.math.exp(-0.1)
input_ids = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'input_ids')
attention_mask = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'attention_mask')
token_type_ids = tf.keras.Input(shape=(max_length,),dtype='int32',name = 'token_type_ids')

output = model_tf([input_ids,attention_mask,token_type_ids])
output = output[0]
output = tf.keras.layers.Dense(128,activation = 'relu')(output)
output = tf.keras.layers.Dropout(0.2)(output)
output = tf.keras.layers.Dense(1,activation = 'softmax')(output)

model = tf.keras.models.Model(inputs = [input_ids,attention_mask,token_type_ids],outputs = output)
# choosing Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.BinaryFocalCrossentropy()
metric = tf.keras.metrics.BinaryAccuracy('acc')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
#callbacks=[early_stopping]
callbacks_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint_filepath = 'modelo_bert_current.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',
    period=3
)
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_history = model.fit(ds_train_encoded, epochs=100, validation_data=ds_test_encoded, callbacks=[callbacks_lr,model_checkpoint_callback])
model.save('modelo_bert_100_2voters.h5')
