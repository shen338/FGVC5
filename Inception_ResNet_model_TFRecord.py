import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
slim = tf.contrib.slim


# total training data number
sample_num = 437513
# State dataset directory where the tfrecord files are located
dataset_dir = '.'

# State where your log file is at. If it doesn't exist, create it.
log_dir = './log'
# tensorboard visualization path
filewriter_path = './filewriter'

# State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'
checkpoint_save_addr = './fine-tuning.ckpt'
# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 299

# State the number of classes to predict:
num_classes = 8142

# ================= TRAINING INFORMATION ==================
# State the number of epochs to train
num_epochs = 10

# State your batch size
batch_size = 10

# Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.8
num_epochs_before_decay = 2


def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Graph().as_default() as graph:

        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        IMAGE_HEIGHT = 299
        IMAGE_WIDTH = 299
        data_path = 'iNaturalist.tfrecords'

        feature = {'train/height': tf.FixedLenFeature([], tf.int64),
                   'train/width': tf.FixedLenFeature([], tf.int64),
                   'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        # create a list of file names
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=num_epochs)

        print(filename_queue)
        reader = tf.TFRecordReader()
        _, tfrecord_serialized = reader.read(filename_queue)

        features = tf.parse_single_example(tfrecord_serialized, features=feature)

        # Convert the image data from string back to the numbers
        height = tf.cast(features['train/height'], tf.int32)
        width = tf.cast(features['train/width'], tf.int32)
        image = tf.decode_raw(features['train/image'], tf.uint8)
        label = tf.cast(features['train/label'], tf.int64)

        image = tf.reshape(image, tf.stack([height, width, 3]))
        # label = tf.reshape(label, [1])

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=IMAGE_HEIGHT,
                                                               target_width=IMAGE_WIDTH)

        images, labels = tf.train.shuffle_batch([resized_image, label], batch_size=batch_size,
                                                capacity=10 * batch_size, num_threads=16,
                                                min_after_dequeue=10, allow_smaller_final_batch=True)

        print("input test")
        print(images)
        print(labels)



        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = sample_num / batch_size
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
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)

        print(one_hot_labels)
        print(logits)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        # RMSProp or Adam
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Create the train_op.
        # this function includes compute_gradients and apply_gradients
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.metrics.accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        writer = tf.summary.FileWriter(filewriter_path)
        writer.add_graph(tf.get_default_graph())

        my_summary_op = tf.summary.merge_all()


        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed
            for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        saver = tf.train.Saver(variables_to_restore)

        # Define your supervisor for running a managed session.
        # Do not run the summary_op automatically or else it will consume too much memory
        with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                               save_checkpoint_secs=600) as sess:

            ckpt = tf.train.get_checkpoint_state(checkpoint_file)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Run the managed session

            for step in range(int(num_steps_per_epoch * num_epochs)):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:

                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    print('logits: \n', logits_value)

                    print('Probabilities: \n', probabilities_value)

                    print('predictions: \n', predictions_value)

                    print('Labels:\n:', labels_value)

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, global_step_count = train_step(sess, train_op, global_step)
                    summaries = sess.run(my_summary_op)
                    writer.add_summary(summaries, global_step_count)
                    # sess.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, global_step)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            saver.save(sess, checkpoint_save_addr, global_step=global_step)



if __name__ == '__main__':
    run()




