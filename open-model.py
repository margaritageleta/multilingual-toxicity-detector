import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

print(tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = tf.keras.models.load_model(f'models/distilbert_batch16_epochs3_maxlen192')
print(model.summary())

with tf.device('cpu:0'):
    model.save(f'models/CPU_distilbert_batch16_epochs3_maxlen192') 

with tf.device('cpu:0'):
    new_model = tf.keras.models.load_model(f'models/CPU_distilbert_batch16_epochs3_maxlen192')
print(new_model.summary())