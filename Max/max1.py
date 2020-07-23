# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dn9iDgozu_6lSPx7RwRsI8TTo1PEv753
"""

#!pip install tf-encrypted

import tensorflow as tf
import tf_encrypted as tfe
import numpy as np

orig_vec = np.array([[0.0, 0.0, 0.32479804907788445, 0.8727328151196464, 0.0, 0.0, 0.0, 0.3610730071635421, 0.0, 0.37540009144947417]])
orig_vec

print(orig_vec.shape)

orig_tensor = tf.convert_to_tensor(orig_vec, dtype=tf.float32)
print(orig_tensor)

#x = []
#for i in range(5000):
#  newarr = np.random.random(10)
#  newarr = np.reshape(newarr, (1, 10))
#  x.append(newarr)

x = []
for i in range(10):
  newtensor = tf.random.uniform((1, 10))
  x.append(newtensor)

#for i in range(len(x)):
#  x[i] = tf.convert_to_tensor(x[i], dtype=tf.float32)

import time

w = tfe.define_private_variable(orig_tensor)

newx = []
for i in range(len(x)):
  newx.append(tfe.define_private_variable(x[i]))


print(w)
print(newx)

y = []
for i in range(len(newx)):
  y.append(tfe.sub(newx[i], w))

print(y)
newy = []
with tfe.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(len(y)):
    newy.append(sess.run(y[i].reveal()))

#print(newy)
newy = np.array(newy).astype(float)

#start = time.time()

with tfe.protocol.SecureNN() as prot:
  start = time.time()
  out_tfe = prot.argmax(prot.define_private_variable(tf.constant(newy)))
  print(time.time() - start)

with tfe.Session() as sess:
  sess.run(tf.global_variables_initializer())
  actual = sess.run(out_tfe.reveal())

