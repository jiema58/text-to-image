
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

def train(batch_size,z_dim):
    tf.reset_default_graph()
    
    real_image=tf.placeholder(dtype=tf.float32,shape=[batch_size,None,None,3])
    wrong_image=tf.placeholder(dtype=tf.float32,shape=[batch_size,None,None,3])
    real_caption=tf.placeholder(dtype=tf.int64,shape=[batch_size,None])
    wrong_caption=tf.placeholder(dtype=tf.int64,shape=[batch_size,None])
    real_caption_length=tf.placeholder(dtype=tf.int64,shape=[batch_size])
    wrong_caption_length=tf.placeholder(dtype=tf.int64,shape=[batch_size])
    latent_z=tf.placeholder(dtype=tf.float32,shape=[None,z_dim])
    lr=tf.placeholder(dtype=tf.float32,shape=[])
    test_caption=tf.placeholder(dtype=tf.int64,shape=[16,None])
    test_caption_length=tf.placeholder(dtype=tf.int64,shape=[16])
    
    model=txt2img()
    
    real_img=process(real_image)
    wrong_img=process(wrong_image)
    
    real_img_code=model.cnn_encoder(real_img)
    wrong_img_code=model.cnn_encoder(wrong_img,reuse=True)
    
    real_caption_code=model.rnn_embed(real_caption,batch_size,real_caption_length)
    wrong_caption_code=model.rnn_embed(wrong_caption,batch_size,wrong_caption_length,reuse=True)
    
    alpha=0.2
    rnn_loss=tf.maximum(0.,tf.reduce_mean(-cosine(real_img_code,real_caption_code)+cosine(wrong_img_code,real_caption_code)))+tf.maximum(0.,tf.reduce_mean(-cosine(real_img_code,real_caption_code)+cosine(real_img_code,wrong_caption_code)))
    
    updated_real_caption_code=model.rnn_embed(real_caption,batch_size,real_caption_length,reuse=True)
    updated_wrong_caption_code=model.rnn_embed(wrong_caption,batch_size,wrong_caption_length,reuse=True)
    test_caption_code=model.rnn_embed(test_caption,16,test_caption_length,reuse=True)
    
    fake_img=model.Generator(latent_z,updated_real_caption_code)
    fake_d=model.Discriminator(fake_img,updated_real_caption_code)
    real_d=model.Discriminator(real_img,updated_real_caption_code,reuse=True)
    wrong_d=model.Discriminator(real_img,updated_wrong_caption_code,reuse=True)
    test_fake=model.Generator(latent_z,test_caption_code,is_train=False,reuse=True)
    
    '''
    d_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_d),logits=real_d))
    d_loss_2=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(wrong_d),logits=wrong_d))
    d_loss_3=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_d),logits=fake_d))
    d_loss=d_loss_1+(d_loss_2+d_loss_3)*0.5
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_d),logits=fake_d))
    '''
    d_loss=-tf.reduce_mean(real_d-0.5*wrong_d-0.5*fake_d)
    g_loss=tf.reduce_mean(-fake_d)
    var_list=tf.trainable_variables()
    
    cnn_var=[var for var in var_list if 'cnn_encoder' in var.name]
    rnn_var=[var for var in var_list if 'rnn_embed' in var.name]
    d_var=[var for var in var_list if 'Discriminator' in var.name]
    g_var=[var for var in var_list if 'Generator' in var.name]
    
    d_optimizer=tf.train.RMSPropOptimizer(lr,name='d_op')
    g_optimizer=tf.train.RMSPropOptimizer(lr,name='g_op')
    rnn_optimizer=tf.train.AdamOptimizer(0.5,name='rnn_op')
    grads,_=tf.clip_by_global_norm(tf.gradients(rnn_loss,rnn_var+cnn_var),10)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_op=d_optimizer.minimize(d_loss,var_list=d_var)
        g_op=g_optimizer.minimize(g_loss,var_list=g_var)
        rnn_op=rnn_optimizer.apply_gradients(zip(grads,rnn_var+cnn_var))
    clip_grad=[p.assign(tf.clip_by_value(p,-0.01,0.01)) for p in d_var]
    saver=tf.train.Saver(tf.global_variables())
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    
    img_size=len(img)
    caption_size=len(caption)
    n_epoch=600
    print_freq=1
    lr_decay=0.5
    decay_every=100
    curr_lr=0.0002
    n_batch=img_size//batch_size
    test_captions,test_length=pad_sequence(test_sample_captions)
    for epoch in range(0,n_epoch+1):
        curr_lr=curr_lr*(lr_decay**(epoch//100))
            
        for step in range(n_batch):
            r_c_idx=np.random.randint(0,caption_size,size=(batch_size))
            r_i_idx=r_c_idx//10
            real_images=img[r_i_idx]
            curr_real_captions=get_caption(r_c_idx)
            w_i_idx,w_c_idx=sample(r_i_idx,img_size,batch_size)
            wrong_images=img[w_i_idx]
            curr_wrong_captions=get_caption(w_c_idx)
            z_sample=np.random.normal(size=(batch_size,z_dim)).astype(np.float32)
            real_captions,real_length=pad_sequence(curr_real_captions)
            wrong_captions,wrong_length=pad_sequence(curr_wrong_captions)
            
            if epoch<50:
                rnn_err,_=sess.run([rnn_loss,rnn_op],feed_dict={latent_z:z_sample,real_image:real_images,wrong_image:wrong_images,real_caption:real_captions,wrong_caption:wrong_captions,real_caption_length:real_length,wrong_caption_length:wrong_length,lr:curr_lr})
            else:
                rnn_err=0
                
            d_err,_,_=sess.run([d_loss,d_op,clip_grad],feed_dict={real_image:real_images,wrong_image:wrong_images,real_caption:real_captions,wrong_caption:wrong_captions,real_caption_length:real_length,wrong_caption_length:wrong_length,latent_z:z_sample,lr:curr_lr})
            g_err,_=sess.run([g_loss,g_op],feed_dict={latent_z:z_sample,real_caption:real_captions,real_caption_length:real_length,lr:curr_lr,wrong_caption_length:wrong_length,real_image:real_images,wrong_image:wrong_images,wrong_caption:wrong_captions})
            
            if epoch>=50 and epoch%100==0 and step%50==0:
                z_sample=np.random.normal(size=(16,z_dim)).astype(np.float32)
                test_img=sess.run(test_fake,feed_dict={latent_z:z_sample,test_caption:test_captions,test_caption_length:test_length})
                plot(test_img)
                
            if epoch%4==0 and step%10==0:
                print('##### Epoch:{}, Step:{} #####\n ***D:{}, G:{}, R:{}***'.format(epoch,step,d_err,g_err,rnn_err))
            
        if epoch>=50 and epoch%10==0:
            saver.save(sess,os.path.join(os.getcwd(),'WGA_model/ga.ckpt'),global_step=epoch)
    saver.save(sess,os.path.join(os.getcwd(),'WGA_model/ga.ckpt'),global_step=epoch)
    sess.close()

train(64,100)