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

sample_size=4
ni=2
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni)+["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni)+["the petals on this flower are white with a yellow center"] * int(sample_size/ni)+["this flower has a lot of small round pink petals."] * int(sample_size/ni)+["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni)+["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni)+["this flower has petals that are blue and white."] * int(sample_size/ni)+["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)
def transfer(sample_sentence):
    res=[]
    for i in sample_sentence:
        i=preprocess_caption(i)
        i=preprocess_sentence(i)
        tmp=[]
        for j in i:
            tmp.append(w2idx[j])
        res.append(tmp)
    return res
test_sample_captions=transfer(sample_sentence)
len(test_sample_captions)


# In[15]:

class txt2img:
    def __init__(self,**param):
        #self.embedding_size=param.get('embedding_size',128)
        self.z_dim=param.get('z_dim',100)
        self.t_dim=param.get('t_dim',128)
        self.vocab_size=param.get('vocab_size',5430)
        self.embed_size=param.get('embed_size',1024)

    def Discriminator(self,input_images,txt,is_train=True,reuse=False):
        d_dim=64
        w_init=tf.random_normal_initializer(stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=0.02)
        with tf.variable_scope('Discriminator',reuse=reuse):
            net_l1=tf.layers.conv2d(input_images,d_dim,4,2,padding='SAME',kernel_initializer=w_init,name='first_layer')
            net_l1=self.lrelu(net_l1)
            
            net_l2=tf.layers.conv2d(net_l1,d_dim*2,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_1')
            #net_l2=tf.layers.batch_normalization(net_l2,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            net_l2=self.instance_norm(net_l2,'in_1')
            net_l2=self.lrelu(net_l2)
            
            net_l3=tf.layers.conv2d(net_l2,d_dim*4,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_2')
            #net_l3=tf.layers.batch_normalization(net_l3,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            net_l3=self.instance_norm(net_l3,'in_2')
            net_l3=self.lrelu(net_l3)
            
            net_l4=tf.layers.conv2d(net_l3,d_dim*8,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_3')
            #net_l4=tf.layers.batch_normalization(net_l4,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            net_l4=self.instance_norm(net_l4,'in_3')
            
            net=self.d_rb(net_l4,d_dim*2,name='d_rb',is_train=is_train)
            net_txt=tf.layers.dense(txt,self.t_dim,kernel_initializer=w_init,name='fc_1')
            net_txt=self.lrelu(net_txt)
            
            exp_txt=tf.expand_dims(tf.expand_dims(net_txt,1),1)
            exp_txt=tf.tile(exp_txt,[1,4,4,1])
            net_concat=tf.concat((net,exp_txt),3)
                
            net=tf.layers.conv2d(net_concat,d_dim*8,1,1,padding='VALID',use_bias=False,kernel_initializer=w_init,name='c_4')
            #net=tf.layers.batch_normalization(net,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_4')
            net=self.instance_norm(net,'in_4')
            net=self.lrelu(net)
                
            net_l5=tf.layers.conv2d(net,1,4,4,padding='VALID',kernel_initializer=w_init,name='final_layer')
            return net_l5
                
    def Generator(self,z,txt,is_train=True,reuse=False):
        g_dim=128
        w_init=tf.random_normal_initializer(stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=.02)
        
        with tf.variable_scope('Generator',reuse=reuse):
            if txt is not None:
                reduced_txt=tf.layers.dense(txt,g_dim,activation=tf.nn.relu,kernel_initializer=w_init,name='input_layer')
                z=tf.concat((z,reduced_txt),1)
            
            net_l1=tf.layers.dense(z,g_dim*8*4*4,use_bias=True,kernel_initializer=w_init,name='fc_1')
            #net_l1=tf.layers.batch_normalization(net_l1,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            #net_l1=self.instance_norm(net_l1,'in_1')
            net_l1=tf.reshape(net_l1,[-1,4,4,g_dim*8])
            
            net_l2=self.g_rb(net_l1,g_dim*2,name='rb_1',is_train=is_train)
            
            net_l3=tf.layers.conv2d_transpose(net_l2,g_dim*4,3,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='dc_1')
            #net_l3=tf.layers.batch_normalization(net_l3,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            net_l3=self.instance_norm(net_l3,'in_2')
            
            net_l4=self.g_rb(net_l3,g_dim,name='rb_2',is_train=is_train)
            
            net_l5=tf.layers.conv2d_transpose(net_l4,g_dim*2,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='dc_2')
            #net_l5=tf.layers.batch_normalization(net_l5,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            net_l5=self.instance_norm(net_l5,'in_3')
            net_l5=tf.nn.relu(net_l5)
            
            net_l6=tf.layers.conv2d_transpose(net_l5,g_dim,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='dc_3')
            #net_l6=tf.layers.batch_normalization(net_l6,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_4')
            net_l6=self.instance_norm(net_l6,'in_4')
            net_l6=tf.nn.relu(net_l6)
            
            net_l7=tf.layers.conv2d_transpose(net_l6,3,4,2,padding='SAME',use_bias=True,kernel_initializer=w_init,name='output_layer')
            out=tf.nn.tanh(net_l7)
            return out
            
    def z_encoder(input_images,is_train=True,reuse=False):
        w_init=tf.random_normal_initializer(mean=1.,stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=0.02)
        d_dim=64
        s=64
        with tf.variable_scope('z_encoder'):
            net_l1=tf.layers.conv2d(input_images,d_dim,4,2,padding='SAME',kernel_initializer=w_init,name='input_layer')
            net_l1=self.lrelu(net_l1)
            
            net_l2=tf.layers.conv2d(net_l1,d_dim*2,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_1')
            net_l2=tf.layers.batch_normalization(net_l2,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            net_l2=self.lrelu(net_l2)
            
            net_l3=tf.layers.conv2d(net_l2,d_dim*4,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_2')
            net_l3=tf.layers.batch_normalization(net_l3,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            net_l3=self.lrelu(net_l3)
            
            net_l4=tf.layers.conv2d(net_l3,d_dim*8,4,2,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_3')
            net_l4=tf.layers.batch_normalization(net_l4,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            net_l4=self.lrelu(net_l4)
            
            net_l5=self.d_rb(net_lr4,d_dim*2,name='z_rb',is_train=is_train)
            
            net_out=tf.reshape(net_l5,[-1,d_dim*8*4*4])
            net_out=tf.layers.dense(net_out,self.z_dim,kernel_initializer=w_init,name='final_layer')
            return net_out
    
    def lrelu(self,x,leak=0.2):
        return tf.maximum(x,x*leak)
    
    def g_rb(self,x,g_dim,name,is_train=True):
        w_init=tf.random_normal_initializer(stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=.02)
        with tf.variable_scope(name):
            c_1=tf.layers.conv2d(x,g_dim,1,1,padding='VALID',use_bias=False,kernel_initializer=w_init,name='c_1')
            #c_1=tf.layers.batch_normalization(c_1,momentum=.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            c_1=self.instance_norm(c_1,'in_1')
            c_1=tf.nn.relu(c_1)
            
            c_2=tf.layers.conv2d(c_1,g_dim,3,1,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_2')
            #c_2=tf.layers.batch_normalization(c_2,momentum=.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            c_2=self.instance_norm(c_2,'in_2')
            c_2=tf.nn.relu(c_2)
            
            c_3=tf.layers.conv2d(c_2,g_dim*4,3,1,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_3')
            #c_3=tf.layers.batch_normalization(c_3,momentum=.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            c_3=self.instance_norm(c_3,'in_3')
            
            return tf.nn.relu(x+c_3)
        
    def d_rb(self,x,d_dim,name,is_train=True):
        w_init=tf.random_normal_initializer(stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=.02)
        with tf.variable_scope(name):
            c_1=tf.layers.conv2d(x,d_dim,1,1,padding='VALID',use_bias=False,kernel_initializer=w_init,name='c_1')
            #c_1=tf.layers.batch_normalization(c_1,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            c_1=self.instance_norm(c_1,'in_1')
            c_1=self.lrelu(c_1)
            c_2=tf.layers.conv2d(c_1,d_dim,3,1,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_2')
            #c_2=tf.layers.batch_normalization(c_2,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            c_2=self.instance_norm(c_2,'in_2')
            c_2=self.lrelu(c_2)
            c_3=tf.layers.conv2d(c_2,d_dim*4,3,1,padding='SAME',use_bias=False,kernel_initializer=w_init,name='c_3')
            #c_3=tf.layers.batch_normalization(c_3,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            c_3=self.instance_norm(c_3,'in_3')
            
            return self.lrelu(x+c_3)
        
    def cnn_encoder(self,inputs,is_train=True,reuse=False):
        w_init=tf.random_normal_initializer(stddev=0.02)
        g_init=tf.random_normal_initializer(mean=1.,stddev=0.02)
        d_dim=64
        
        with tf.variable_scope('cnn_encoder',reuse=reuse):
            net_l1=tf.layers.conv2d(inputs,d_dim,4,2,padding='SAME',kernel_initializer=w_init,name='input_layer')
            net_l1=self.lrelu(net_l1)
            
            net_l2=tf.layers.conv2d(net_l1,d_dim*2,4,2,padding='SAME',use_bias=True,kernel_initializer=w_init,name='c_1')
            #net_l2=tf.layers.batch_normalization(net_l2,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_1')
            #net_l2=self.instance_norm(net_l2,'in_1')
            net_l2=self.lrelu(net_l2)
            
            net_l3=tf.layers.conv2d(net_l2,d_dim*4,4,2,padding='SAME',use_bias=True,kernel_initializer=w_init,name='c_2')
            #net_l3=tf.layers.batch_normalization(net_l3,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_2')
            #net_l3=self.instance_norm(net_l3,'in_2')
            net_l3=self.lrelu(net_l3)
            
            net_l4=tf.layers.conv2d(net_l3,d_dim*8,4,2,padding='SAME',use_bias=True,kernel_initializer=w_init,name='c_3')
            #net_l4=tf.layers.batch_normalization(net_l4,momentum=0.9,gamma_initializer=g_init,training=is_train,name='bn_3')
            #net_l4=self.instance_norm(net_l4,'in_3')
            net_l4=self.lrelu(net_l4)
            
            net_l5=tf.reshape(net_l4,[-1,d_dim*8*4*4])
            net_l5=tf.layers.dense(net_l5,self.t_dim,kernel_initializer=w_init,name='final_layer')
            
        return net_l5
    
    def rnn_embed(self,inputs,batch_size,sequence_length,reuse=False):
        w_init=tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope('rnn_embed',initializer=w_init,reuse=reuse):
            cell=tf.contrib.rnn.BasicLSTMCell(self.t_dim)
            embedding=tf.get_variable(name='rnn/embedding',dtype=tf.float32,shape=[self.vocab_size,self.embed_size])
            embed_out=tf.nn.embedding_lookup(embedding,inputs)
            out,(c_state,h_state)=tf.nn.dynamic_rnn(cell,embed_out,sequence_length=sequence_length,initial_state=cell.zero_state(batch_size,tf.float32),scope='dynamic_rnn')
            return h_state
        
    def instance_norm(self,x,name='i_n'):
        b_init=tf.constant_initializer(0.)
        g_init=tf.random_normal_initializer(mean=1.,stddev=.02)
        with tf.variable_scope(name):
            mean,var=tf.nn.moments(x,axes=[1,2],keep_dims=True)
            standard_x=(x-mean)/tf.sqrt(var+1e-5)
            beta=tf.get_variable('i_n_beta',shape=[x.get_shape()[-1]],dtype=tf.float32,initializer=b_init)
            gamma=tf.get_variable('i_n_gamma',shape=[x.get_shape()[-1]],dtype=tf.float32,initializer=g_init)
            out=standard_x*gamma+beta
            return out

