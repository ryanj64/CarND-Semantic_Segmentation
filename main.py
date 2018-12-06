#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np
import scipy

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from video import process_video


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the model and weights into tensorflow
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    model = tf.get_default_graph()
    vgg_input_tensor = model.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = model.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = model.get_tensor_by_name(
        vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = model.get_tensor_by_name(
        vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = model.get_tensor_by_name(
        vgg_layer7_out_tensor_name)

    return (vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor)


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # 1 x 1 convolution
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    layer8 = tf.layers.conv2d(vgg_layer7_out,
                              num_classes,
                              1,
                              strides=(1, 1),
                              padding='same',
                              kernel_regularizer=regularizer,
                              kernel_initializer=k_initializer,
                              bias_initializer=b_initializer)

    # Create a deconvolusion on the 8th layer with an output size to match the VGG 4th pooling layer.
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    layer9_0 = tf.layers.conv2d_transpose(layer8,
                                          num_classes,
                                          4,
                                          strides=(2, 2),
                                          padding='same',
                                          kernel_regularizer=regularizer,
                                          kernel_initializer=k_initializer,
                                          bias_initializer=b_initializer)

    # layer4 is scaled before its usage
    pool4_out_scaled = tf.multiply(vgg_layer4_out,
                                   0.01,
                                   name='pool4_out_scaled')

    # 1 x 1 convolution of the scaled 4th layer
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    layer9_1 = tf.layers.conv2d(pool4_out_scaled,
                                num_classes,
                                1,
                                strides=(1, 1),
                                padding='same',
                                kernel_regularizer=regularizer,
                                kernel_initializer=k_initializer,
                                bias_initializer=b_initializer)

    # Add a skip connection
    layer9 = tf.add(layer9_0, layer9_1)

    # Create a deconvolusion on the 9th layer with an output size to match the VGG 3rd layer.
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    layer10_0 = tf.layers.conv2d_transpose(layer9,
                                           num_classes,
                                           4,
                                           strides=(2, 2),
                                           padding='same',
                                           kernel_regularizer=regularizer,
                                           kernel_initializer=k_initializer,
                                           bias_initializer=b_initializer)

    # layer3 is scaled before its usage
    pool3_out_scaled = tf.multiply(vgg_layer3_out,
                                   0.0001,
                                   name='pool3_out_scaled')

    # 1 x 1 convolution
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    layer10_1 = tf.layers.conv2d(pool3_out_scaled,
                                 num_classes,
                                 1,
                                 strides=(1, 1),
                                 padding='same',
                                 kernel_regularizer=regularizer,
                                 kernel_initializer=k_initializer,
                                 bias_initializer=b_initializer)

    # Add a skip connection
    layer10 = tf.add(layer10_0, layer10_1)

    # Create a deconvolusion on the 10th layer (output).
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    k_initializer = tf.random_normal_initializer(stddev=1e-2)
    b_initializer = tf.zeros_initializer()
    output = tf.layers.conv2d_transpose(layer10,
                                        num_classes,
                                        16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_regularizer=regularizer,
                                        kernel_initializer=k_initializer,
                                        bias_initializer=b_initializer)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Flatten the tensor to a one dimensional array
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    # Adjustable variables
    lr_value = 0.0001
    kp_value = 0.70

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Extract batch features and labels.
        for batch_features, batch_labels in get_batches_fn(batch_size):
            # Run the optimizer
            sess.run(train_op, feed_dict={input_image: batch_features,
                                          correct_label: batch_labels, keep_prob: kp_value, learning_rate: lr_value})

        # Run the loss function
        train_loss = sess.run(cross_entropy_loss, feed_dict={input_image: batch_features,
                                                             correct_label: batch_labels, keep_prob: 1.0})

        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_constant = 1e-3.
        # loss_w_reg = train_loss + reg_constant * tf.reduce_sum(reg_losses)
        loss_w_reg = tf.add(train_loss, tf.losses.get_regularization_loss())

        # Print out the Epoch and Loss value.
        print('Epoch {:>2}, Loss: {:>10.4f}, Loss with Reg: {:>10.4f}'.format(epoch, 
            train_loss,  
            tf.Tensor.eval(loss_w_reg)))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    epochs = 40
    batch_size = 8

    # Placeholders
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(
        tf.float32, [None, image_shape[0], image_shape[1], num_classes], name="correct_label")

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road', 'training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_tensor, keep_prob_tensor, L3_tensor_out, L4_tensor_out, L7_tensor_out = load_vgg(
            sess, vgg_path)

        nn_last_layer = layers(
            L3_tensor_out, L4_tensor_out, L7_tensor_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(
            nn_last_layer, correct_label, learning_rate, num_classes)

        # Write graph to a file for display in tensorboard.
        logs_dir = os.path.join('.', 'graphs')
        if not os.path.exists(logs_dir):
            os.mkdir('graphs')
        # writer = tf.summary.FileWriter(logs_dir, sess.graph)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_tensor, correct_label, keep_prob_tensor, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        runs_dir = os.path.join('.', 'results')
        if not os.path.exists(runs_dir):
            os.mkdir('results')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob_tensor, input_tensor)

        # Save Model
        save_filename = os.path.join(os.sep, 'SS')
        save_path = os.path.join('.', 'checkpoints')
        if not os.path.exists(save_path):
            os.mkdir('checkpoints')

        saver = tf.train.Saver()
        saver.save(sess, save_path + save_filename)
        print('Model Trained and Saved')

        # close the writer when you’re done using it
        # writer.close()

    # OPTIONAL: Apply the trained model to a video
    # Created a seperate pyhton script to do this.
    vid = process_video()
    vid.setup(save_path + save_filename,
              image_shape,
              'image_input:0',
              'keep_prob:0',
              'logits:0')
    vid.run()


if __name__ == '__main__':
    run()
