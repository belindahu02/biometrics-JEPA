from transformations import *
import tensorflow as tf
import numpy as np

def flip_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Flip(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_flip(input):
  y = tf.numpy_function(flip_numpy, [input], tf.float64)
  return y
  
def scale_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_scale(input):
  y = tf.numpy_function(scale_numpy, [input], tf.float64)
  return y
  
def jitter_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Jitter(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_jitter(input):
  y = tf.numpy_function(jitter_numpy, [input], tf.float64)
  return y
  
def magwarp_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_MagWarp(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_magwarp(input):
  y = tf.numpy_function(magwarp_numpy, [input], tf.float64)
  y = tf.ensure_shape(y, (input.shape[0], input.shape[1]))  # (frames, features)

  # Optionally cast to float32 for compatibility with model
  y = tf.cast(y, tf.float32)
  return y
  
def timewarp_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_TimeWarp(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_timewarp(input):
  y = tf.numpy_function(timewarp_numpy, [input], tf.float64)
  return y
  
def permutation_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Permutation(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_permutation(input):
  y = tf.numpy_function(permutation_numpy, [input], tf.float64)
  return y
  
def randsampling_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_RandSampling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_randsampling(input):
  y = tf.numpy_function(randsampling_numpy, [input], tf.float64)
  return y
  
def negation_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Negation(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_negation(input):
  y = tf.numpy_function(negation_numpy, [input], tf.float64)
  return y
  
def chf_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_ChannelShuffle(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_chf(input):
  y = tf.numpy_function(chf_numpy, [input], tf.float64)
  return y
  
def drop_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Drop(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_drop(input):
  y = tf.numpy_function(drop_numpy, [input], tf.float64)
  return y
  
def random_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  #augs = [DA_Jitter, DA_Flip, DA_Scaling, DA_MagWarp, DA_Drop]
  #indx = np.arange(len(augs))
  #np.random.shuffle(indx)
  #aug = augs[indx[0]]
  x = DA_Scaling(x)
  return x
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_random(input):
  y = tf.numpy_function(random_numpy, [input], tf.float64)
  return y