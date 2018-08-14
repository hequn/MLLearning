import numpy as np
import tensorflow as tf
import random as rn
from keras.layers import multiply,concatenate,Embedding
from keras.layers.merge import dot
from keras import backend as K
from keras.models import Sequential


# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

x1 = np.array([]).reshape(0,4)
x1 = np.append(x1,np.array([1,2,3,4]).reshape(1,4),axis=0)
x1 = np.append(x1,np.array([3,4,5,6]).reshape(1,4),axis=0)
x1 = np.append(x1,np.array([5,6,7,8]).reshape(1,4),axis=0)

y1 = np.array([]).reshape(0,4)
y1 = np.append(y1,np.array([7,8,9,10]).reshape(1,4),axis=0)
y1 = np.append(y1,np.array([9,10,11,12]).reshape(1,4),axis=0)
y1 = np.append(y1,np.array([11,12,13,14]).reshape(1,4),axis=0)

print(x1-y1)

x = tf.placeholder(tf.float64, [3, 4])
y = tf.placeholder(tf.float64, [3, 4])
labels = tf.placeholder(tf.float64, [256])

xxx = K.sum(K.square(x-y),1,keepdims=True)
yyy = dot([x,K.transpose(y)],(0,1))
zzz = tf.matmul(tf.transpose(x,perm=[0,1]),tf.transpose(y,perm=[1,0]))
hhh = multiply([x,y])
labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
labels_not_equal = tf.logical_not(labels_equal)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    xxxx = sess.run(xxx, feed_dict={x:x1,y:y1})
    print(xxxx)
    yyyy = sess.run(yyy, feed_dict={x:x1,y:y1})
    print(yyyy)
    zzzz = sess.run(zzz, feed_dict={x:x1,y:y1})
    print(zzzz)
    hhhh = sess.run(hhh, feed_dict={x:x1,y:y1})
    print(hhhh)
    labels_test = sess.run(labels_equal, feed_dict={labels:np.random.randint(256, size=(256))})
    labels_test_not_equal = sess.run(labels_not_equal, feed_dict={labels:np.random.randint(256, size=(256))})
    print(labels_test)
# Rest of code follows ...

# x = K.variable(value=x1)
# y = K.variable(value=y1)
#
# z = K.dot(x,K.transpose(y))
#
# # Here you need to use K.eval() instead of z.eval() because this uses the backend session
# print(K.eval(z))


# x_batch = K.ones(shape=(32, 20, 1))
# y_batch = K.ones(shape=(32, 30, 20))
# xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
# print(K.int_shape(xy_batch_dot))

#Lambda(lambda x: K.batch_dot(x, x, axes=(2, 2)), output_shape=lambda s: (s[0], s[1], s[1]))

# def multiply(x,n):
#     x_prime = tf.reshape(x, (-1, n, 1))
#     x_transpose = tf.transpose(x_prime, perm=[0,2, 1])
#     return tf.batch_matmul(x_transpose,x_prime)
# Lambda(lambda x: multiply(x, n), output_shape =(n, n))

model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)