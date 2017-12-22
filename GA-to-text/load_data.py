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
from utils import *

caption,caption_list,word=collect_caption('D:/dataset/text-to-image/text_c10')
w2idx,vocab=build_vocab(word,1)
caption_ids=transfer_caption(caption_list,w2idx)
#img=read_img('D:/dataset/jpg')

metadata={}
metadata['w2idx']=w2idx
metadata['idx2w']=vocab
#save_all(img,'img.pickle')
save_all(metadata,'metadata.pickle')
save_all(caption_ids,'caption.pickle')


with open('metadata.pickle','rb') as f:
    metadata=pickle.load(f)
w2idx=metadata['w2idx']
idx2w=metadata['idx2w']


img=load_img()
caption=load_caption()
print('img size: %d, caption size: %d'%(len(img),len(caption)))

print('vocab size: ',len(idx2w))