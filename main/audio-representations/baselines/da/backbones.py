from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Add, GlobalMaxPooling1D, MaxPooling1D, Dropout

def resnetblock(inputs, KS, CR, skip=True):
    conv1 = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    sum_ = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(conv2)

    if skip:
        conv4 = Conv1D(filters=CR ,kernel_size=1, strides=1, padding='same')(inputs)
        sum_ = Add()([sum_, conv4])

    sum_ = BatchNormalization()(sum_)
    sum_ = ReLU()(sum_)
    outputs = MaxPooling1D(pool_size=4, strides=4)(sum_)

    return sum_ 
    
def resnetblock_final(inputs, KS, CR, skip=True):
    conv1 = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    sum_ = Conv1D(filters=CR ,kernel_size=KS, strides=1, padding='same')(conv2)

    if skip:
        conv4 = Conv1D(filters=CR ,kernel_size=1, strides=1, padding='same')(inputs)
        sum_ = Add()([sum_, conv4])

    sum_ = BatchNormalization()(sum_)
    sum_ = ReLU()(sum_)
    outputs = GlobalMaxPooling1D()(sum_)

    return outputs