import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Activation,BatchNormalization, Add
import numpy as np


#define SeConv block:
class SeConv_block(keras.layers.Layer):
    def __init__(self, kernel_size, input_channels, **kwargs):
        super(SeConv_block, self).__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels


    def build(self, input_shape):
        kernel_init = tf.ones_initializer()
        self.kernel = tf.Variable(name="kernel", initial_value=kernel_init(shape=(self.kernel_size, self.kernel_size, self.input_channels, 1), dtype='float32'),trainable=True)
        

    def call(self, inputs):
        # non-noisy pixels map:
        M_hat = tf.math.not_equal(inputs, 0)
        M_hat = tf.dtypes.cast(M_hat, tf.float32)
        
        
        conv_input = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=None, name=None)
        conv_M_hat = tf.nn.conv2d(M_hat, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=None, name=None)
        
        # find 0 in conv_M_hat and change to 1:
        is_zero_conv_M_hat = tf.equal(conv_M_hat, 0)
        is_zero_conv_M_hat = tf.dtypes.cast(is_zero_conv_M_hat, tf.float32)
        change_zero_to_one_conv_M_hat = tf.math.add(conv_M_hat, is_zero_conv_M_hat)

        S = tf.math.divide(conv_input, change_zero_to_one_conv_M_hat)


        # noisy pixels map:
        M = 1 - M_hat

        # calculate R:
        kernel_ones = np.ones((self.kernel_size, self.kernel_size, self.input_channels, 1))
        kernel_ones = tf.constant(kernel_ones, dtype=tf.float32)
        
        R = tf.nn.conv2d(M_hat, kernel_ones, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=None, name=None)
        R = tf.math.greater_equal(R, tf.constant(self.kernel_size-2, dtype=tf.float32))
        R = tf.dtypes.cast(R, tf.float32)

        y = tf.math.multiply(tf.math.multiply(S, M), R) + inputs

        return y

    def get_config(self):
        config = super(SeConv_block, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


#___________________________________________________________________


# define model:
def SeConvNet(num_SeConv_block=7,depth=27,filters=64,image_channels=1):
  layer_count = 0
  inputs = Input(shape=(None,None,image_channels), name='input'+str(layer_count))
  
  # 1st to 7th layers (SeConv_block):
  x = inputs
  for i in range(num_SeConv_block):
      layer_count += 1
      x = SeConv_block(2*layer_count+1, image_channels, name='SeConv_block'+str(layer_count))(x)
  

  # 8th to 26th layers, Conv+BN+ReLU:
  for i in range(depth-num_SeConv_block-1):
       layer_count += 1
       x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal', padding='same', use_bias = False, name='Conv'+str(layer_count))(x)
       x = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='BN'+str(layer_count))(x) 
       x = Activation('relu', name='ReLU'+str(layer_count))(x)  
  
  
  # last layer, Conv:
  layer_count =+ 1    
  x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal', padding='same', use_bias = False, name='Conv'+str(layer_count))(x)
  
  
  x =  keras.layers.Multiply(name='Multiply')([x, tf.dtypes.cast(tf.math.equal(inputs, 0), tf.float32)])

  outputs = Add(name='Add')([x, inputs])

  model = keras.models.Model(inputs=inputs, outputs=outputs)
  return model