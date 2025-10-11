import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from backbones import *
from data_loader import *

#0 #17
# single layer: ft=5 #12
# 2 layer: ft=6 #11
# 3 layer: ft=9 #8
# 4 layer: ft=12 #5
# all layer: ft=17 #0

def trainer(samples_per_user, fet_extrct, scen, ft):
    
  ft_dict = {0:17, 1:12, 2:11, 3:8, 4:5, 5:0}
  ft = ft_dict[ft]
  
  #fet_extrct.trainable = False
  for i in range(1,ft+1):
    fet_extrct.layers[i].trainable = False
  
  frame_size   = 30
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/musicid_dataset/"
  
  users_2 = list(range(7,21)) #Users for dataset 2
  users_1 = list(range(1,7)) #Users for dataset 1
  
  ######################################################Transfering##########################################################################################
  
  folder_train = ["TrainingSet"]
  folder_val = ["TestingSet"]
  folder_test = ["TestingSet_secret"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_2, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  x_val, y_val, sessions_val = data_load_origin(path, users=users_2, folders=folder_val, frame_size=30)
  print("validation samples : ", x_val.shape[0])
  
  x_test, y_test, sessions_test = data_load_origin(path, users=users_2, folders=folder_test, frame_size=30)
  print("testing samples : ", x_test.shape[0])
  
  classes, counts  = np.unique(y_train, return_counts=True)
  num_classes = len(classes)
  print("minimum samples per user : ", min(counts)) #60
  
  x_train, x_val, x_test = norma(x_train, x_val, x_test)
  print("x_train", x_train.shape)
  print("x_val", x_val.shape)
  print("x_test", x_test.shape)
  
  x_train, y_train = user_data_split(x_train ,y_train , samples_per_user=samples_per_user)
  print("limited training samples : ", x_train.shape[0])
  classes, counts  = np.unique(y_train, return_counts=True)
  print(counts)
  
  #fet_extrct=model.layers[len(transformations)]

  #fet_extrct.trainable=False
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = fet_extrct(inputs, training=False)
  x = Dense(256, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(num_classes, activation='softmax')(x)
  resnettssd = Model(inputs, outputs)
  #resnettssd.summary()
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  #optimizer = tf.keras.optimizers.Adam()
  resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
  history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=8)
  
  results = resnettssd.evaluate(x_test,y_test)
  test_acc = results[1]
  print("test acc:", results[1])
  
  #Calculating kappa score
  metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
  metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
  result = metric.result()
  kappa_score = result.numpy()
  print('kappa score: ',result.numpy())
  
  return test_acc, kappa_score