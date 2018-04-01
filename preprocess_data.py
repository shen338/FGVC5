#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:15:04 2018

"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import Augmentor
import keras
import tensorflow
import random
from skimage import transform

def load_train_data(min_dir,sub_dir):
    dir_all=os.path.join(min_dir,sub_dir)
    with open(dir_all) as load_f:
        load_dict=json.load(load_f)
        annotations=load_dict['annotations']
        images=load_dict['images']

    imgs_dir=[]
    img_label=[]
    img_super_label=[]
    sup_lab={'Plantae':0,'Insecta':1,'Aves':2,'Actinopterygii':3,'Fungi':4,'Reptilia':5
             ,'Mollusca':6,'Mammalia':7,'Animalia':8,'Amphibia':9,'Arachnida':10,
             'Chromista':11,'Protozoa':12,'Bacteria':13}
    for i in range (len(images)):
        img_dir=images[i]['file_name']
        imgs_dir.append(os.path.join(min_dir,img_dir))
        img_lab=annotations[i]['category_id']
        img_label.append(img_lab)
        sup_name=img_dir.split('/')[1].split('/')[0]
        img_super_label.append(sup_lab[sup_name])
    return imgs_dir,np.array(img_label),np.array(img_super_label)


def load_test_data(min_dir,sub_dir):
     dir_all=os.path.join(min_dir,sub_dir)
     with open(dir_all) as load_f:
         load_dict=json.load(load_f)
         images=load_dict['images']
     imgs_dir=[]
     for i in range (len(images)):
        img_dir=images[i]['file_name']
        imgs_dir.append(os.path.join(min_dir,img_dir))
     return imgs_dir
        

        
#data augmentation
def data_aug(imgs, labels, batch_size, scaled=True, image_data_format="channels_last"):
    imgs=np.uint8(imgs*255)
    P=Augmentor.Pipeline()
    P.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    P.flip_left_right(probability=0.4)
    P.flip_top_bottom(probability=0.2)
    P.rotate90(probability=0.1)
    P.random_distortion(probability=1,grid_width=10,grid_height=10,magnitude=1)
    P.skew_corner(probability=0.3)
    P.shear(probability=0.3,max_shear_left=10,max_shear_right=10)
    g=P.keras_generator_from_array(imgs,labels,batch_size=batch_size)
    return g
#    g=P.keras_generator_from_array

def data_aug_v2(imgs, labels, batch_size, scaled=True, image_data_format="channels_last"):
    # imgs=np.uint8(imgs)
    P=Augmentor.Pipeline()
    P.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    P.flip_left_right(probability=0.4)
    P.flip_top_bottom(probability=0.4)
    P.rotate90(probability=0.1)
    P.random_distortion(probability= 1,grid_width=10,grid_height=10,magnitude=1)
    P.skew_corner(probability=0.3)
    P.shear(probability=0.3,max_shear_left=10,max_shear_right=10)
    g=P.keras_generator_from_array(imgs,labels,batch_size=batch_size)
    images, labels=next(g)
    return images, labels

def generate_batch_data(sup_classes, sup_lab, imgs_dir, input_sup_classes, batch_size,):
    train_imgs=[]
    train_labels=[]
    train_sup_labels=[]
    for i in range (sup_classes):
        one_cls_list=np.argwhere(sup_lab==i)
        sup_list=random.sample(list(one_cls_list),input_sup_classes)
        for k in sup_list:
            img_dir=imgs_dir[k[0]] 
            one_img=mpimg.imread(os.path.join(img_dir))
            one_img=transform.resize(one_img,(299,299,3))
            train_imgs.append(one_img)
            # train_labels.append(lab[k])
            train_sup_labels.append(sup_lab[k])
    train_imgs=np.array(train_imgs)
    g=data_aug(train_imgs,train_sup_labels,batch_size=batch_size)
    X,y=next(g) 
    return X,y    

if __name__=='__main__':
    '''
    choose several data from 14 sup_classes and generate batch size data with 
    rotaton and so on ,each time randomely select images from different classes
    ''' 
##training data
    min_dir='/home/pami3/Downloads/Kaggle_data'
    sub_dir='train2018.json'
    imgs_dir,lab,sup_lab=load_train_data(min_dir,sub_dir)
 ##test_data
    test_log_dir='test2018.json'
    test_dir=load_test_data(min_dir,test_log_dir)

    batch_size=64  #for training , we select batch_size*4 images from different sup_classes
    iteration=1000
    sup_classes=14
    input_data=batch_size*4
    input_sup_classes=min(input_data//sup_classes,16)
    for count in range(iteration):
        batch_x,batch_y=generate_batch_data(sup_classes,sup_lab,imgs_dir,input_sup_classes)
        #the network for train...
        
    '''
     X,y is the data and sup_label from different super_classes, we randomely augmented the input images 
     and choose part of them as training batches, then put them back into pooling and re-select, which
     promise the balance of different classes and differents from batches to batches.
    '''

    
        
        
    
        
        
        
        
        
        
