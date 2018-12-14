# -*- coding: utf-8 -*-

"""Apply style transfer to a given image"""

# Import standard library
import os
import glob
import logging

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import coloredlogs
from magenta.models.arbitrary_image_stylization import (
    arbitrary_image_stylization_build_model as build_model,
)
from magenta.models.image_stylization import image_utils

# Define slim graphmaker
slim = tf.contrib.slim


class Styler:
    """Styles a set of images (also called content) given a certain style"""

    def __init__(self, checkpoint, output):
        """Initialize the class

        checkpoint: str
            Path to the model checkpoint (model.ckpt)
        output : str
            Path for output files
        """
        self.checkpoint = checkpoint
        self.output = output
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(logging.INFO, logger=self.logger)
        if not tf.gfile.Exists(output):
            tf.gfile.MkDir(output)

    def _create_placeholder(self, square_crop, img_ph, size):
        """Create input image placeholder

        Returns a subgraph with resize steps for the image

        Parameters
        ----------
        square_crop : bool
            Perfom square cropping or not
        img_ph : tf.Tensor
            Input image placeholder

        Returns
        -------
        tf.Tensor
           Image placeholder with preprocessing step
        """
        if square_crop:
            self.logger.debug('Cropping image')
            img_preprocessed = image_utils.center_crop_resize_image(
                img_ph, size
            )
        else:
            img_preprocessed = image_utils.resize_image(img_ph, size)
        return img_preprocessed

    def _get_img_lists(self, content_path, style_path, max_styles):
        """Get content and style image lists

        Parameters
        ----------
        content_path : str
            Path to content images
        style_path : str
            Path to style images
        max_styles : int
            Maximum number of styles

        Returns
        -------
        (list, list)
            Content image list and style image list
        """
        content_img_list = glob.glob('./**/{}'.format(content_path), recursive=True)
        style_img_list = glob.glob('./**/{}'.format(style_path), recursive=True)
        if len(style_img_list) == 0 or len(content_img_list) == 0:
            msg = 'Content or style path not found! Make sure it\'s in repo root!'
            self.logger.error(msg)
            raise ValueError(msg)

        if len(style_img_list) > max_styles:
            self.logger.warn(
                'Image list is greater than max_styles_to_evaluate'
            )
            np.random.seed(1234)
            style_img_list = np.random.permutation(style_img_list)
            style_img_list = style_img_list[:max_styles]
        return (content_img_list, style_img_list)

    def _get_data_and_name(self, img_path):
        """Get data as numpy array and name of file

        Parameters
        ----------
        img_path : str
            Path to target image

        Returns
        -------
        (np.ndarray, str)
            N-dim array representation and filename
        """
        img_np = image_utils.load_np_image_uint8(img_path)[:, :, :3]
        img_name = os.path.basename(img_path)[:-4]
        return (img_np, img_name)

    def _save_preprocessed_img(
        self, sess, img_preprocessed, img_ph, img_np, img_name
    ):
        """Save preprocessed image when necessary

        Parameters
        ----------
        sess : tf.Session
            Session is required to perform compute
        img_preprocessed: tf.Tensor
            Tensor graph for preprocessing
        img_ph: tf.Tensor
            Tensor input placeholder
        img_np: np.ndarray
            N-dimensional array with data
        img_name: string
            Image file name
        """
        # Create a numpy array of the cropped image
        img_cropped_resized_np = sess.run(
            img_preprocessed,  # graph for preprocessing images
            feed_dict={img_ph: img_np},  # numpy input
        )
        output_path = os.path.join(self.output, '{}.jpg'.format(img_name))
        image_utils.save_np_image(img_cropped_resized_np, output_path)
        self.logger.debug('Image saved at {}'.format(output_path))

    def _save_image(self, img, output_file):
        self.logger.info('Saving stylized image at: {}'.format(output_file))
        img = np.uint8(img * 255.0)
        plt.imsave(output_file, img, dpi=300)

    def _run_tf_graph(
        self,
        sess,
        content_path,
        style_path,
        content_size,
        style_size,
        content_square_crop,
        style_square_crop,
        interp_weights,
        max_styles_to_evaluate,
    ):
        """Create a Tensorflow static graph that defines all computation needed

        Parameters
        ----------
        sess : tf.Session
            Session to run this graph on
        content_path : str
            Path to content images
        style_path : str
            Path to style images
        content_size : int
            Size to resize the content image into
        style_size : int
            Size to resize the style image into
        content_square_crop : bool
            Trigger to crop the content into a square
        style_square_crop : bool
            Trigger to crop the style into a square
        interp_weights : list
            Interpolation weights
        max_styles_to_evaluate : int
            Maximum number of styles to run style transfer into
        """
        # Define placeholder for style image
        style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        style_img_preprocessed = self._create_placeholder(
            style_square_crop, style_img_ph, style_size
        )
        # Define placeholder for content image
        content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        content_img_preprocessed = self._create_placeholder(
            content_square_crop, content_img_ph, content_size
        )
        # Define the model
        stylized_images, _, _, bottleneck_feat = build_model.build_model(
            content_img_preprocessed,
            style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False,
        )

        # Load checkpoint
        if tf.gfile.IsDirectory(self.checkpoint):
            checkpoint = tf.train.latest_checkpoint(self.checkpoint)
        else:
            checkpoint = self.checkpoint
            self.logger.info(
                'Loading latest checkpoint file: {}'.format(checkpoint)
            )

        init_fn = slim.assign_from_checkpoint_fn(
            model_path=checkpoint, var_list=slim.get_variables_to_restore()
        )
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

        # Get list of content and style images
        content_img_list, style_img_list = self._get_img_lists(
            content_path, style_path, max_styles_to_evaluate
        )

        for content_i, content_img_path in enumerate(content_img_list):
            content_img_np, content_img_name = self._get_data_and_name(
                img_path=content_img_path
            )

            # Compute bottleneck features of the style prediction network
            # for the identity transform
            identity_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: content_img_np}
            )

            for style_i, style_img_path in enumerate(style_img_list):
                if style_i > max_styles_to_evaluate:
                    break
                style_img_np, style_img_name = self._get_data_and_name(
                    img_path=style_img_path
                )

                if style_i % 10 == 0:
                    self.logger.info(
                        'Stylizing ({}) {} with {} {}'.format(
                            content_i,
                            content_img_name,
                            style_i,
                            style_img_name,
                        )
                    )

                # Compute bottleneck features of the style prediction
                style_params = sess.run(
                    bottleneck_feat, feed_dict={style_img_ph: style_img_np}
                )
                for interp_i, wi in enumerate(interp_weights):
                    stylized_image_res = sess.run(
                        stylized_images,
                        feed_dict={
                            bottleneck_feat: identity_params * (1 - wi)
                            + style_params * wi,
                            content_img_ph: content_img_np,
                        },
                    )

                    # Save stylized image
                    fname = os.path.join(
                        self.output,
                        '{}_stylized_{}_{}.png'.format(
                            content_img_name, style_img_name, interp_i
                        ),
                    )
                    self._save_image(stylized_image_res[0], fname)

    def style_transfer(
        self,
        content_path,
        style_path,
        content_size=400,
        style_size=256,
        content_square_crop=False,
        style_square_crop=True,
        interp_weights=[1.0],
        max_styles_to_evaluate=1024,
    ):
        """Create a Tensorflow static graph that defines all computation needed

        Parameters
        ----------
        content_path : str
            Path to content images
        style_path : str
            Path to style images
        content_size : int (default is 256)
            Size to resize the content image into
        style_size : int (default is 256)
            Size to resize the style image into
        content_square_crop : bool (default is False)
            Trigger to crop the content into a square
        style_square_crop : bool (default is False)
            Trigger to crop the style into a square
        interp_weights : list (default is [1.0])
            Interpolation weights
        max_styles_to_evaluate : int (default is 1024)
            Maximum number of styles to run style transfer into
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            self._run_tf_graph(
                sess,
                content_path=content_path,
                style_path=style_path,
                content_size=content_size,
                style_size=style_size,
                content_square_crop=content_square_crop,
                style_square_crop=style_square_crop,
                interp_weights=interp_weights,
                max_styles_to_evaluate=max_styles_to_evaluate,
            )
