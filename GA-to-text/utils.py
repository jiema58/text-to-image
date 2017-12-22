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

def preprocess_caption(line):
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('_', ' ')
    return prep_line

def preprocess_sentence(sentence,start_word='<S>',end_word='<\S>'):
    #process_sentence=[start_word]
    process_sentence=nltk.tokenize.word_tokenize(sentence.lower())
    #process_sentence.append(end_word)
    return process_sentence

def collect_caption(path):
    caption={}
    caption_list={}
    word=[]
    for (dirpath,dirname,filenames) in os.walk(path):
        if os.path.isdir(dirpath):
            for filename in filenames:
                if filename.endswith('.txt') or filename.endswith('.TXT'):
                   #print(filename)
                    key=int(re.findall('\d+',filename)[0])
                    file_path=os.path.join(dirpath,filename)
                    lines=[]
                    lines_=[]
                    with open(file_path,'r') as f:
                        for line in f:
                            line=preprocess_caption(line)
                            line_=preprocess_sentence(line)
                            lines.append(line)
                            lines_.append(line_)
                            word.extend(line_)
                    caption_list[key]=lines_
                    caption[key]=lines
                    assert len(lines)==10
    return caption,caption_list,word
                    
def build_vocab(word,min_word_count):
    counter=Counter(word)
    word_counts=[x for x in counter.items() if x[1]>=min_word_count]
    word_counts.sort(key=lambda x:x[1],reverse=True)
    word_counts=[('<PAD>',0)]+word_counts
    print('total words: %d in vocabulary'%len(word_counts))
    reserved_vocab=[x[0] for x in word_counts]+['unk']
    w2idx=dict(zip(reserved_vocab,range(len(reserved_vocab))))
    return w2idx,reserved_vocab

def transfer_caption(caption_list,w2idx):
    caption_ids={}
    for i in caption_list.keys():
        tmp=[]
        for j in caption_list[i]:
            tmp.append([w2idx[k] for k in j])
        caption_ids[i]=tmp
    return caption_ids

def read_img(path):
    img_pool={}
    path_list=glob(path+'/*.jpg')
    for i in path_list:
        base_path=os.path.basename(i)
        key=int(re.findall('\d+',base_path)[0])
        img=scipy.misc.imread(i)
        img=scipy.misc.imresize(img,[79,79],interp='bilinear')
        img_pool[key]=img
    return img_pool

def save_all(targets,file):
    with open(file,'wb') as f:
        pickle.dump(targets,f)

def pad_sequence(sequence,max_len=None):
    curr_len=[len(i) for i in sequence]
    if not max_len:
        max_len=max(curr_len)
    res=np.zeros((len(sequence),max_len))
    for i in range(len(sequence)):
        res[i,0:len(sequence[i])]=sequence[i]
    return res,curr_len

def load_caption(path=None,to_list=True):
    if not path:
        path=os.getcwd()+'/caption.pickle'
    with open(path,'rb') as f:
        caption_ids=pickle.load(f)
    caption=[]
    if to_list:
        for i in range(1,len(caption_ids)+1):
            caption.extend(caption_ids[i])
        return caption
    else:
        return caption_ids
    
def load_img(path=None):
    if not path:
        path=os.getcwd()+'/img.pickle'
    print(path)
    with open(path,'rb') as f:
        img=pickle.load(f)
    image=[]
    for i in range(len(img)):
        image.append(img[i+1])
    return np.array(image).astype(np.float32)/127.5-1.

def process(img,batch_size=None):
    if not batch_size:
        batch_size=img.get_shape().as_list()[0]
    img=tf.random_crop(img,[batch_size,64,64,3])
    images=tf.split(img,batch_size,0)
    for i in range(batch_size):
        images[i]=tf.expand_dims(tf.image.random_flip_left_right(tf.squeeze(images[i],0)),0)
    return tf.concat(images,0)
                                 
def image_batch(image_path,batch_size,epoch):
    image_path=image_path+'/*.jpg'
    image_path=glob(image_path)
    reader=tf.WholeFileReader()
    img_queue=tf.train.string_input_producer(image_path,num_epochs=epoch)
    _,img=reader.read(img_queue)
    img=tf.image.decode_image(img,channels=3)
    img.set_shape([79,79,3])
    img=tf.image.random_flip_left_right(img)
    img=tf.image.resize_images(img,[64+15,64+15])
    img==tf.random_crop(img,[64,64,3])
    img=tf.cast(img,tf.float32)/127.5-1.
    
    min_after_dequeue=batch_size*2 
    capacity=batch_size*6+min_after_dequeue
   
    img_batch=tf.train.batch([img],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
    return img_batch             
                                 
def cosine(x,y):
    cost=tf.reduce_sum(x*y,1)/(tf.sqrt(tf.reduce_sum(x*x,1))*tf.sqrt(tf.reduce_sum(y*y,1)))
    return cost

def sample(idx,i,batch_size):
    img_idx=[]
	    caption_idx=[]
    for _ in range(batch_size):
        j=idx[0]
        while j in idx:
            j=np.random.randint(0,i)
        img_idx.append(j)
    for _ in range(batch_size):
        j=idx[0]
        while j in idx:
            j=np.random.randint(0,i)
        caption_idx.append(j*10+np.random.randint(0,10))
    return img_idx,caption_idx

def plot(test_img):
    test_img=(test_img+1.)*127.5
    test_img=test_img.astype(np.uint8)
    f,axs=plt.subplots(8,2,figsize=(20,12))
    for i in range(8):
        for j in range(2):
            axs[i][j].imshow(test_img[i*2+j])
            axs[i][j].axis('off')
    plt.show()
    
def get_caption(idx):
    res=[]
    for i in idx:
        res.append(caption[i])
    return res