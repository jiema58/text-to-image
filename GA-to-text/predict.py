import numpy as np
import tensorflow as tf
import nltk
import string
from collections import Counter
import os
import pickle
import scipy.misc
import matplotlib.pyplot as plt
from glob import glob
import re
from untils import *
from model import *

def predict(batch_size,inputs=None):
    z=tf.placeholder(dtype=tf.float32,shape=[batch_size,100])
    caption=tf.placeholder(dtype=tf.float32,shape=[batch_size,None])
    length=tf.placeholder(dtype=tf.int64,shape=[batch_size])
     model=txt2img()
    caption_ids=model.rnn_embed(caption,batch_size,length)
    img=model.Generator(z,caption,is_train=False)
    saver=tf.train.Saver()
    test_captions,test_length=pad_sequence(test_sample_captions)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,tf.train.latest_checkpoint('GA_model'))
        images=sess.run(img,{z:np.random.normal(size=(batch_size,100)),caption:sa})
        plot(images)
