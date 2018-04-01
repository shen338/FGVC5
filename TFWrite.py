import tensorflow as tf
import cv2
import json
import numpy as np


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


folder = '/home/tong/Downloads/'
annotation_dir = '/home/tong/Downloads/train2018.json'
filename = 'iNaturalist_test.tfrecords'


with open(annotation_dir) as json_data:
    data = json.load(json_data)
    print('annotation length: ', len(data['annotations']))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




# create new TFrecord file
writer = tf.python_io.TFRecordWriter(filename)

super_label = 0
for i in range(200):
    image_id = data['annotations'][i]['image_id']
    image_dir = data['images'][image_id]['file_name']
    image = open(folder + image_dir, 'rb').read()
    label = data['annotations'][i]['category_id']
    height = data['images'][image_id]['height']
    width = data['images'][image_id]['width']

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

writer.close()
