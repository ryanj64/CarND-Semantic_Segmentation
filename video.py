#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np
import scipy

# Import everything needed to edit/save video clips
from moviepy.editor import VideoFileClip


class process_video:
    def __init__(self):
        self.save_model_path = None
        self.image_shape = None

        # Load model
        self.loader = None
        self.session = None

        # Get Tensors from loaded model
        self.loaded_inputs = None
        self.loaded_keep_prob = None
        self.loaded_logits = None

    def setup(self, model_path, image_shape, input_tensor_name, keep_prob_tensor_name, logits_tensor_name):
        # Reset the graph
        tf.reset_default_graph()

        self.save_model_path = model_path  # os.path.join('.', 'checkpoints', 'SS')
        self.image_shape = image_shape  # (160, 576)

        # Create session
        self.session = tf.Session()
        # Load model
        self.loader = tf.train.import_meta_graph(self.save_model_path + '.meta')
        # Restore
        self.loader.restore(self.session, self.save_model_path)
        # Get Tensors from loaded model
        self.loaded_inputs = self.session.graph.get_tensor_by_name(input_tensor_name)
        self.loaded_keep_prob = self.session.graph.get_tensor_by_name(keep_prob_tensor_name)
        self.loaded_logits = self.session.graph.get_tensor_by_name(logits_tensor_name)

    def pipeline(self, image):

        img = scipy.misc.imresize(image, self.image_shape)
        # img = np.expand_dims(img, axis=0)

        with self.session.as_default():

            # Print names from graph (for debugging only)
            # graph_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            # print(*graph_names, sep='\n')

            im_softmax = self.session.run([tf.nn.softmax(self.loaded_logits)],
                                          feed_dict={self.loaded_inputs: [img], self.loaded_keep_prob: 1.0})

            im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(img)
            street_im.paste(mask, box=None, mask=mask)

        return np.array(street_im)

    def run(self):
        # OPTIONAL: Apply the trained model to a video
        if (self.save_model_path != None and
            self.image_shape != None and
            self.loader != None and
            self.session != None and
            self.loaded_inputs != None and
            self.loaded_keep_prob != None and
            self.loaded_logits != None):
            # Load video
            clip = VideoFileClip('driving.mp4')

            new_clip = clip.fl_image(self.pipeline)

            # Write to file
            new_clip.write_videofile('result.mp4')
        else:
            print('Video could not be processed due to missing information! Did you run setup() before run()?')
