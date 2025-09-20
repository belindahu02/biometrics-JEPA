from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Conv1DTranspose, Dropout, GaussianNoise

from tensorflow.keras import regularizers

def predexp(inputs, mlp_s=2048):

    x = Dense(mlp_s//2, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    #x = BatchNormalization()(x)
    outputs = Dense(mlp_s//2, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    return outputs
    
def predTian(inputs, mlp_s=256):
    #x = GaussianNoise(stddev=0.01)(inputs)
    x = Dense(mlp_s*4)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.35)(x)  
    x = Dense(mlp_s*2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.35)(x)
    x = Dense(mlp_s)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    outputs = Dense(mlp_s//4)(x)
    
    return outputs


def predTian_Origin(inputs, mlp_s=256):
    x = Dense(mlp_s, kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    outputs = Dense(mlp_s//4, kernel_regularizer=regularizers.l2(0.01))(x)
    
    return outputs