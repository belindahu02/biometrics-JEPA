from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

from backbones import *
from data_loader import *
from transformations_tf import *

def compute_cohen_kappa(y_true, y_pred, num_classes):
  """
  Manual Cohen's Kappa implementation (no tensorflow_addons).
  """
  y_true = np.array(y_true).astype(int)
  y_pred = np.argmax(y_pred, axis=1).astype(int)

  confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
  for t, p in zip(y_true, y_pred):
    confusion[t, p] += 1

  n_samples = np.sum(confusion)
  po = np.trace(confusion) / n_samples
  pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (n_samples ** 2)

  kappa = (po - pe) / (1 - pe + 1e-12)
  return kappa

def trainer(samples_per_user):
  frame_size   = 30
  BATCH_SIZE = 8
  AUTO = tf.data.AUTOTUNE
  path = "/app/1.0.0/"
  # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"

  users_2 = list(range(1, 110))  # Users for dataset 2
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
  
  SEED = 34
  ds_x = tf.data.Dataset.from_tensor_slices(x_train)
  #ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
  ds_x = (
      ds_x.shuffle(1024, seed=SEED)
      .map(tf_magwarp, num_parallel_calls=AUTO)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )

  for batch in ds_x.take(1):
    print("ds_x batch shape:", batch.shape)
  
  ds_y = tf.data.Dataset.from_tensor_slices(y_train)
  ds_y = (
      ds_y.shuffle(1024, seed=SEED)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )

  for batch in ds_y.take(1):
    print("ds_y batch shape:", batch.shape)

  ssl_ds = tf.data.Dataset.zip((ds_x, ds_y))
  
  val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
  
  ks = 3
  con =3
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling1D(pool_size=4, strides=4)(x)
  x = Dropout(rate=0.1)(x)
  x = resnetblock_final(x, CR=32*con, KS=ks)
  x = Flatten()(x)
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
  history = resnettssd.fit(ssl_ds, validation_data=val_ds, epochs=100, callbacks=callback, batch_size=BATCH_SIZE)
  
  results = resnettssd.evaluate(x_test,y_test)
  test_acc = results[1]
  print("test acc:", results[1])
  
  #Calculating kappa score
  # metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
  # metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
  # result = metric.result()
  # kappa_score = result.numpy()
  y_pred_probs = resnettssd.predict(x_test)
  kappa_score = compute_cohen_kappa(y_test, y_pred_probs, num_classes)
  print('kappa score: ', kappa_score)
  
  return test_acc, kappa_score