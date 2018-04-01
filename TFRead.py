import tensorflow as tf
import cv2
import numpy as np
import time
from preprocess_data import data_aug_v2
from test_imgaug import augment
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299

data_path = ['iNaturalist_0.tfrecords', 'iNaturalist_1.tfrecords',
             'iNaturalist_2.tfrecords', 'iNaturalist_3.tfrecords',
             'iNaturalist_4.tfrecords', 'iNaturalist_5.tfrecords',
             'iNaturalist_6.tfrecords', 'iNaturalist_7.tfrecords',
             'iNaturalist_8.tfrecords', 'iNaturalist_9.tfrecords',
             'iNaturalist_10.tfrecords', 'iNaturalist_11.tfrecords',
             'iNaturalist_12.tfrecords', 'iNaturalist_13.tfrecords']


batch_size = 42
cate_size = int(batch_size/14)

feature = {'train/height': tf.FixedLenFeature([], tf.int64),
           'train/width': tf.FixedLenFeature([], tf.int64),
           'train/image': tf.FixedLenFeature([], tf.string),
           'train/label': tf.FixedLenFeature([], tf.int64),
           'train/sup_label': tf.FixedLenFeature([], tf.int64)}

tf_images = []
tf_labels = []
print('length: ', sum(1 for _ in tf.python_io.tf_record_iterator(data_path[7])))
filename_queue = []
for ii in range(14):

    # create a list of file names
    filename_queue = tf.train.string_input_producer([data_path[ii]], num_epochs=None)

    print(filename_queue)

    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(filename_queue)

    features = tf.parse_single_example(tfrecord_serialized, features=feature)

    # Convert the image data from string back to the numbers
    height = tf.cast(features['train/height'], tf.int32)
    width = tf.cast(features['train/width'], tf.int32)

    # change this line for your TFrecord version
    # image = tf.image.decode_raw(features['train/image'], tf.uint8)
    tf_image = tf.image.decode_jpeg(features['train/image'])

    tf_label = tf.cast(features['train/label'], tf.int64)
    tf_sup_label = tf.cast(features['train/sup_label'], tf.int64)

    tf_image = tf.reshape(tf_image, tf.stack([height, width, 3]))
    # tf_image = tf.cast(tf_image, tf.uint8)
    #label = tf.reshape(label, [1])

    resized_image = tf.image.resize_images(images=tf_image, size=tf.constant([400, 400]), method=2)
    resized_image = tf.cast(resized_image, tf.uint8)

    images_temp, labels_temp = tf.train.shuffle_batch([resized_image, tf_sup_label], batch_size=cate_size,
                                                capacity=1024, num_threads=32,
                                                min_after_dequeue=256, allow_smaller_final_batch=False)

    tf_images.append(images_temp)
    tf_labels.append(labels_temp)

tf_images = tf.concat(tf_images, axis=0)
tf_labels = tf.concat(tf_labels, axis=0)

with tf.Session() as sess:

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(10000):

        if batch_index %50 == 0:
            print(batch_index)
            print(time.clock())


        # here img lbl are numpy array of a batch
        # equivalent to generate_batch but without augmentation
        imgs, lbls = sess.run([tf_images, tf_labels])

        # print(imgs.shape)

        # imgs, lbls = data_aug_v2(imgs, lbls, batch_size)

        imgs = imgs[:, 50:349, 50:349, :]

        # img_aug = augment(img)
        # img_aug, lbl = data_aug_v2(img, lbl, 42)
        # here is your augment function(numpy array version)
        # img = data_aug(img, lbl, batch_size)

        # print(time.clock())

        # print(img.shape)
        # print(lbl.shape)

        # img = img.astype(np.uint8)

        # print(time.clock())
        # print(imgs.shape)
        # print(lbls)
        # for jj in range(14):
        #     # img[jj,:,:,:] = cv2.cvtColor(img[jj,:,:,:], cv2.COLOR_RGB2BGR)
        #     plt.imshow(imgs[jj,:,:,:])
        #     plt.show()
        #     print(imgs[jj, 1:10,1:10, 1])

        # cv2.waitKey(0)

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()

