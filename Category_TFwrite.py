import tensorflow as tf
import cv2
import json
import numpy as np
import glob
import os


def load_image(addr):
    '''
    :param addr: image directory
    :return: image with uint8 data type
    '''
    img = cv2.imread(addr)
    # img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    return img

sup_lab={'Plantae':0,'Insecta':1,'Aves':2,'Actinopterygii':3,'Fungi':4,'Reptilia':5
             ,'Mollusca':6,'Mammalia':7,'Animalia':8,'Amphibia':9,'Arachnida':10,
             'Chromista':11,'Protozoa':12,'Bacteria':13}

folder = '/home/tong/Downloads/'
annotation_dir = '/home/tong/Downloads/train2018.json'
directory = '/home/tong/cate8/'

with open(annotation_dir) as json_data:
    data = json.load(json_data)
    print('annotation length: ', len(data['annotations']))

length = len(data['annotations'])

for super in range(14):

    # if super != 7:
    #     continue

    filename = 'iNaturalist_' + str(super) + '.tfrecords'
    super_label = super

    count = 0

    # cate = data['annotations'][1]['category_id']
    # sup_cate = data['categories'][cate]['supercategory']
    #
    # print(sup_lab[sup_cate])

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    # create new TFrecord file
    writer = tf.python_io.TFRecordWriter(filename)


    for i in range(length):
        cate = data['annotations'][i]['category_id']
        sup_cate = data['categories'][cate]['supercategory']
        if sup_lab[sup_cate] == super:

            image_dir = data['images'][i]['file_name']
            # image = load_image(folder + image_dir)
            # print(folder + directory + image_dir)
            # if not os.path.exists(os.path.dirname(directory + image_dir)):
            #     os.mkdir(os.path.dirname(directory + image_dir))
            # cv2.imwrite(folder + image_dir, image)
            image = open(folder + image_dir, 'rb').read()
            label = cate
            height = data['images'][i]['height']
            width = data['images'][i]['width']

            feature = {'train/height': _int64_feature(height),
                       'train/width': _int64_feature(width),
                       'train/image': _bytes_feature(image),
                       'train/label': _int64_feature(label),
                       'train/sup_label': _int64_feature(super_label)
                       }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

            count = count + 1
            if count % 50 == 0:
                print('TF super category progress: ', super)
                print('TF record progress: ', count)

    writer.close()
