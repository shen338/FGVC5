import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
from preprocess_data import data_aug_v2
import numpy as np

slim = tf.contrib.slim

# total training data number
sample_num = 437513

reg_scale = 1e-4

# # data folder
# min_dir = '/home/tong/Downloads'
#
# image_dir = 'train_val2018'
# label_dir = 'train2018.json'

data_path = ['iNaturalist_0.tfrecords', 'iNaturalist_1.tfrecords',
             'iNaturalist_2.tfrecords', 'iNaturalist_3.tfrecords',
             'iNaturalist_4.tfrecords', 'iNaturalist_5.tfrecords',
             'iNaturalist_6.tfrecords', 'iNaturalist_7.tfrecords',
             'iNaturalist_8.tfrecords', 'iNaturalist_9.tfrecords',
             'iNaturalist_10.tfrecords', 'iNaturalist_11.tfrecords',
             'iNaturalist_12.tfrecords', 'iNaturalist_13.tfrecords',]

# State where your log file is at. If it doesn't exist, create it.
log_dir = './log_v3'
# tensorboard visualization path
filewriter_path = './filewriter_v3_finetuning_Logits'

# State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'
checkpoint_save_addr = './log_v3/fine-tuning_v3.ckpt'
# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 299

# State the number of classes to predict:
num_classes = 14

# ================= TRAINING INFORMATION ==================
# State the number of epochs to train
num_epochs = 5

# State your batch size
batch_size = 42

# Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0005
learning_rate_decay_factor = 0.8
num_epochs_before_decay = 2


def run():
    # training data
    # imgs_dir, lab, sup_lab = load_train_data(min_dir, label_dir)
    #
    # sup_classes = 14
    # input_data = batch_size * 4
    # input_sup_classes = min(input_data // sup_classes, 16)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    cate_size = int(batch_size / 14)

    feature = {'train/height': tf.FixedLenFeature([], tf.int64),
               'train/width': tf.FixedLenFeature([], tf.int64),
               'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64),
               'train/sup_label': tf.FixedLenFeature([], tf.int64)}

    tf_images = []
    tf_labels = []

    for ii in range(14):

        # create a list of file names
        filename_queue = tf.train.string_input_producer([data_path[ii]], num_epochs=None)

        print(filename_queue)
        reader = tf.TFRecordReader()
        _, tfrecord_serialized = reader.read(filename_queue)

        features = tf.parse_single_example(tfrecord_serialized, features=feature)

        # Convert the image data from string back to the numbers
        height = tf.cast(features['train/height'], tf.int64)
        width = tf.cast(features['train/width'], tf.int64)

        # change this line for your TFrecord version
        # image = tf.image.decode_raw(features['train/image'], tf.uint8)
        tf_image = tf.image.decode_jpeg(features['train/image'])

        tf_label = tf.cast(features['train/label'], tf.int64)
        tf_sup_label = tf.cast(features['train/sup_label'], tf.int64)

        tf_image = tf.reshape(tf_image, tf.stack([height, width, 3]))
        tf_sup_label = tf.reshape(tf_sup_label, [1])

        resized_image = tf.image.resize_images(images=tf_image, size=tf.constant([400, 400]), method=2)
        resized_image = tf.cast(resized_image, tf.uint8)
        images_temp, labels_temp = tf.train.shuffle_batch([resized_image, tf_sup_label], batch_size=cate_size,
                                                          capacity=1024, num_threads=32, allow_smaller_final_batch=False,
                                                          min_after_dequeue=256)

        tf_images.append(images_temp)
        tf_labels.append(labels_temp)

    tf_images = tf.concat(tf_images, axis=0)
    tf_labels = tf.concat(tf_labels, axis=0)

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    IMAGE_HEIGHT = 299
    IMAGE_WIDTH = 299

    images = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
    labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    # Know the number steps to take before decaying the learning rate and batches per epoch
    num_batches_per_epoch = int(sample_num / batch_size)
    num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

    # Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=num_classes, is_training=True)

    # Define the scopes that you want to exclude for restoration
    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    print("label test")
    print(labels)
    print(logits)

    # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
    one_hot_labels = tf.squeeze(tf.one_hot(labels, num_classes), [1])

    print(one_hot_labels)
    print(logits)

    # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)

    # aux_loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=end_points['AuxLogits'],
    #                                            scope='Aux_loss')

    # reg_loss = reg_scale*tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    total_loss = tf.losses.get_total_loss()

    # D_loss = tf.add_n(D_loss, name='Digit_loss')

    # A_loss = tf.add(aux_loss, reg_loss, name='Aux_loss')

    # A_loss = tf.add_n(A_loss, name='Aux_loss')

    # total_loss = digit_loss + 0.3 * aux_loss + reg_loss

    # Create the global step for monitoring the learning_rate and training.
    global_step = tf.train.get_or_create_global_step()

    # Define your exponentially decaying learning rate
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    # train_vars = []
    # Now we can define the optimizer that takes on the learning rate
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          "InceptionResnetV2/Logits")

    # train_vars.extend(train_vars_digits)
    # train_vars_aux = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                    "InceptionResnetV2/AuxLogits")
    # RMSProp or Adam
    # train_vars.extend(train_vars_aux)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Create the train_op.
    # this function includes compute_gradients and apply_gradients
    train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=train_vars)

    # train_op = optimizer.minimize(total_loss, var_list=train_vars)

    # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
    predictions = tf.argmax(end_points['Predictions'], 1)
    probabilities = end_points['Predictions']
    accuracy, accuracy_update = tf.metrics.accuracy(predictions, labels)
    metrics_op = tf.group(accuracy_update, probabilities)

    # Now finally create all the summaries you need to monitor and group them into one summary op.
    # tf.summary.scalar('losses/Total_Loss', total_loss)
    # tf.summary.scalar('losses/Reg_Loss', reg_loss)
    # tf.summary.scalar('losses/Digit_Loss', loss)
    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', lr)
    writer = tf.summary.FileWriter(filewriter_path)
    writer.add_graph(tf.get_default_graph())

    my_summary_op = tf.summary.merge_all()

    def train_step(sess, train_op, global_step, imgs, lbls):
        '''
        Simply runs a session for the three arguments provided and gives a logging on the time elapsed
        for each global step
        '''
        # Check the time for each sess run
        start_time = time.time()

        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op],
                                                    feed_dict={images: imgs, labels: lbls})

        time_elapsed = time.time() - start_time

        # Run the logging to print some results
        logging.info('global step %s: digit_loss: %.4f (%.2f sec/step)',
                     global_step_count, total_loss, time_elapsed)

        return total_loss, global_step_count

    saver = tf.train.Saver(variables_to_restore)
    saver_train = tf.train.Saver(train_vars)

    # Define your supervisor for running a managed session.
    # Do not run the summary_op automatically or else it will consume too much memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # config = tf.ConfigProto(gpu_options=gpu_options)

    config = tf.ConfigProto(allow_soft_placement=True)
    # 最多占gpu资源的70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # 开始不会给tensorflow全部gpu资源 而是按需增加
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, checkpoint_file)

        for step in range(int(num_steps_per_epoch * num_epochs)):

            # At the start of every epoch, show the vital information:
            start_time = time.time()

            imgs, lbls = sess.run([tf_images, tf_labels])

            imgs, lbls = data_aug_v2(imgs, lbls, batch_size)

            imgs = imgs[:, 50:349, 50:349, :]

            imgs = 2*(imgs.astype(np.float32)) - 1

            # print('mean: ', np.mean(imgs))

            lbls = lbls.astype(np.int32)

            print('loading time: ', time.time() - start_time)

            # print(imgs[0, 0:10, 0:10, 0:2])

            if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)

                learning_rate_value, accuracy_value = sess.run([lr, accuracy],
                                                               feed_dict={images: imgs, labels: lbls})

                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)

                # optionally, print your logits and predictions for a sanity check that things are going fine.
                logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                    [logits, probabilities, predictions, labels],
                    feed_dict={images: imgs, labels: lbls})

                print('logits: \n', logits_value)

                print('Probabilities: \n', probabilities_value)

                print('predictions: \n', predictions_value)

                print('Labels:\n:', labels_value)

            # Log the summaries every 10 step.
            if step % 20 == 0:

                loss, global_step_count = train_step(sess, train_op, global_step, imgs, lbls)

                summaries = sess.run(my_summary_op, feed_dict={images: imgs, labels: lbls})

                writer.add_summary(summaries, global_step_count)
                # sess.summary_computed(sess, summaries)

            # If not, simply run the training step

            else:
                loss, _ = train_step(sess, train_op, global_step, imgs, lbls)

            if step % 2000 == 0:

                logging.info('Saving model to disk now.')
                saver_train.save(sess, checkpoint_save_addr, global_step=global_step)

        # We log the final training loss and accuracy
        logging.info('Final Loss: %s', loss)
        logging.info('Final Accuracy: %s', sess.run(accuracy))

        # Once all the training has been done, save the log files and checkpoint model
        logging.info('Finished training! Saving model to disk now.')
        saver.save(sess, checkpoint_save_addr, global_step=global_step)

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run()




