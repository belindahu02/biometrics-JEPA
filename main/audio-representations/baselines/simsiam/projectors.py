from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Conv1DTranspose, Dropout

from tensorflow.keras import regularizers

def proexp(inputs, mlp_s=2048):

  x = BatchNormalization()(inputs)
  x = Dense(mlp_s//8, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  x = Dense(mlp_s//4, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  outputs = layers.Dense(mlp_s//2, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
  
  return outputs
  
def proTian(inputs, mlp_s=256):

  x = Dense(mlp_s, kernel_regularizer=regularizers.l2(0.01), use_bias=False)(inputs)#0.0001
  x = BatchNormalization()(x)
  x = ReLU()(x)
  #x = Dropout(0.35)(x)
  x = Dense(mlp_s//4, kernel_regularizer=regularizers.l2(0.1), use_bias=False)(x)#0.0001 # best 0.01
  outputs = BatchNormalization()(x)
  
  return outputs