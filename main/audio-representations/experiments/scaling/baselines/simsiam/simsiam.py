import pandas as pd
from math import gcd
import numpy as np
from backbones import *
from projectors import *
from predictors import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import Flatten

def get_encoder(frame_size,ftr,mlp_s,origin):
    # Input and backbone.
    ks = 3
    con = 3
    inputs = layers.Input((frame_size,ftr))
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32*con, KS=ks)
    
    outputs = proTian(x,mlp_s=mlp_s)
    
    return tf.keras.Model(inputs, outputs, name="encoder")
    

def get_predictor(mlp_s, origin):
    
    inputs = layers.Input((mlp_s//4,))
    
    if origin:
        outputs = predTian_Origin(inputs, mlp_s=mlp_s)
    else:
        outputs = predTian(inputs, mlp_s=mlp_s)
    
    return tf.keras.Model(inputs, outputs, name="predictor")
    

def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    #print(p.shape,z.shape)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

class Contrastive(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(Contrastive, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            #loss = compute_loss(p1, GaussianNoise(stddev=5)(z2)) / 2 + compute_loss(p2, GaussianNoise(stddev=5)(z1)) / 2
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        