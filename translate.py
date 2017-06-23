from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--input_dir_B", help="path to folder containing images")
parser.add_argument("--image_height", help="image height")
parser.add_argument("--image_width", help="image width")
parser.add_argument("--model", required=True, choices=["pix2pix", "pix2pix2", "CycleGAN"])
parser.add_argument("--generator", default="unet", choices=["unet", "resnet", "highwaynet", "densenet"])
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--restore", default="model", choices=["all", "generators"])
parser.add_argument("--untouch", default="nothing", choices=["nothing", "core"], help="excluded from training")
parser.add_argument("--loss", default="log", choices=["log", "square"])
parser.add_argument("--gen_loss", default="fake", choices=["fake", "negative", "contra"])
parser.add_argument("--X_type", default="image",  choices=["image", "label"])
parser.add_argument("--Y_type", default="image",  choices=["image", "label"])

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--classic_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

if a.image_height is None:
    a.image_height = CROP_SIZE
if a.image_width is None:
    a.image_width = CROP_SIZE


Examples = collections.namedtuple("Examples", "input_paths, target_paths, inputs, targets, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_classic, gen_grads_and_vars, train")
Pix2Pix2Model = collections.namedtuple("Pix2Pix2Model", "predict_real_X, predict_fake_X, predict_real_Y, predict_fake_Y, discrim_X_loss, discrim_Y_loss, discrim_X_grads_and_vars, discrim_Y_grads_and_vars, gen_G_loss_GAN, gen_F_loss_GAN, gen_G_loss_classic, gen_F_loss_classic, gen_G_grads_and_vars, gen_F_grads_and_vars, outputs, reverse_outputs, train")
CycleGANModel = collections.namedtuple("CycleGANModel", "predict_real_X, predict_fake_X, predict_real_Y, predict_fake_Y, discrim_X_loss, discrim_Y_loss, discrim_X_grads_and_vars, discrim_Y_grads_and_vars, gen_G_loss_GAN, gen_F_loss_GAN, forward_cycle_loss_classic, backward_cycle_loss_classic, gen_G_grads_and_vars, gen_F_grads_and_vars, outputs, reverse_outputs, train, cycle_consistency_loss_classic")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)


def conv(batch_input, out_channels, size=4, stride=2, initializer=tf.random_normal_initializer(0, 0.02)):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [size, size, in_channels, out_channels], dtype=tf.float32, initializer=initializer)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        p = int((size - 1) / 2)
        padded_input = tf.pad(batch_input, [[0, 0], [p, p], [p, p], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def noise(input, std):
    gaussian_noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
    return input + gaussian_noise


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# synchronize seed for image operations so that we do the same augmentation operations to both
# input and output images, but only if not CycleGAN
seed_for_random_cropping_X = random.randint(0, 2 ** 31 - 1)
seed_for_random_cropping_Y = random.randint(0, 2 ** 31 - 1) if a.model == "CycleGAN" else seed_for_random_cropping_X


def transform(image, seed):
    r = image
    if a.mode == 'train':  # augment image by flipping and cropping
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        width = tf.shape(image)[1]  # [height, width, channels]
        height = tf.shape(image)[0]  # [height, width, channels]

        # resize when image too small to crop, otherwise use original full image
        r = tf.cond(tf.logical_or(width < a.scale_size, height < a.scale_size),
                    lambda: tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA),
                    lambda: r)

        # offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        # if a.scale_size > CROP_SIZE:
        #     r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        # elif a.scale_size < CROP_SIZE:
        #     raise Exception("scale size cannot be less than crop size")
        r = tf.random_crop(r, size=[CROP_SIZE, CROP_SIZE, 3], seed=seed)

        r.set_shape([CROP_SIZE, CROP_SIZE, 3])  # must do this if tf.image.resize is not used, otherwise shape unknown

    else:  # use full sized original image
        r.set_shape([a.image_height, a.image_width, 3])  # use full size image

    return r


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    if a.input_dir_B is None:   # image pair A and B
        n_images, a_paths, raw_image = load_images(a.input_dir, 'AB')
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_image)[1] # [height, width, channels]
        a_images = preprocess(raw_image[:,:width//2,:])
        b_images = preprocess(raw_image[:,width//2:,:])
        b_paths = a_paths
        print("examples count = %d (each A and B)" % n_images)

    elif not os.path.exists(a.input_dir_B):  # images B in other directory
        raise Exception("input_dir_B does not exist")
    else:  # load A and B images
        n_a_images, a_paths, raw_a_image = load_images(a.input_dir, 'A')
        a_images = preprocess(raw_a_image)
        n_b_images, b_paths, raw_b_image = load_images(a.input_dir_B, 'B')
        b_images = preprocess(raw_b_image)
        print("examples count = %d, %d (A, B)" % (n_a_images, n_b_images))
        n_images = max(n_a_images, n_b_images)

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
        input_paths, target_paths = [a_paths, b_paths]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
        input_paths, target_paths = [b_paths, a_paths]
    else:
        raise Exception("invalid direction")

    with tf.name_scope("input_images"):
        input_images = transform(inputs, seed=seed_for_random_cropping_X)

    with tf.name_scope("target_images"):
        target_images = transform(targets, seed=seed_for_random_cropping_Y)

    if a.model == "CycleGAN":  # unpaired_images
        input_paths_batch, inputs_batch = tf.train.batch([input_paths, input_images], batch_size=a.batch_size, name="input_batch")
        target_paths_batch, targets_batch = tf.train.batch([target_paths, target_images], batch_size=a.batch_size, name="target_batch")
    else:  # paired images
        input_paths_batch, target_paths_batch, inputs_batch, targets_batch = \
            tf.train.batch([input_paths, target_paths, input_images, target_images], batch_size=a.batch_size, name="paired_batch")

    steps_per_epoch = int(math.ceil(n_images / a.batch_size))

    return Examples(
        input_paths=input_paths_batch,
        target_paths=target_paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        steps_per_epoch=steps_per_epoch,
    )


def load_images(input_dir, input_name=''):
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png
    if len(input_paths) == 0:
        raise Exception("%s contains no images (jpg/png)" % input_dir)
    else:
        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)

        with tf.name_scope("load_%simages" % input_name):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = decode(contents)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input.set_shape([None, None, 3])

    return len(input_paths), paths, raw_input


def create_u_net(generator_inputs, generator_outputs_channels):

    max_depth = 8
    ngf = a.ngf * np.array([1, 2, 4, 8, 8, 8, 8, 8])

    def encoder_decoder(input, depth):
        if depth==max_depth:
            return input

        with tf.variable_scope("encoder_%d" % depth):
            down = lrelu(input, 0.2)
            down = conv(down, ngf[depth], stride=2)
            down = batchnorm(down)

        up = encoder_decoder(down, depth + 1)

        with tf.variable_scope("decoder_%d" % depth):
            output = tf.concat([up, down], axis=3)
            output = tf.nn.relu(output)
            output = deconv(output, ngf[depth])
            output = batchnorm(output)
            if depth > 5:
                output = tf.nn.dropout(output, keep_prob=0.5)

        return output

    with tf.variable_scope("encoder_1"):  # [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        down = conv(generator_inputs, ngf[1], stride=2)

    up = encoder_decoder(down, 2)

    with tf.variable_scope("decoder_1"):  # [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        output = tf.concat([up, down], axis=3)
        output = tf.nn.relu(output)
        output = deconv(output, generator_outputs_channels)
        output = tf.tanh(output)

    return output

#
# def create_u_net(generator_inputs, generator_outputs_channels):
#     layers = []
#
#     # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
#     with tf.variable_scope("encoder_1"):
#         output = conv(generator_inputs, a.ngf, stride=2)
#         layers.append(output)
#
#     layer_specs = [
#         a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
#         a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
#         a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
#         a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
#         a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
#         a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
#         a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
#     ]
#
#     for out_channels in layer_specs:
#         with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
#             rectified = lrelu(layers[-1], 0.2)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
#             convolved = conv(rectified, out_channels, stride=2)
#             output = batchnorm(convolved)
#             layers.append(output)
#
#     layer_specs = [
#         (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
#         (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
#         (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
#         (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
#         (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
#         (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
#         (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
#     ]
#
#     num_encoder_layers = len(layers)
#     for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
#         skip_layer = num_encoder_layers - decoder_layer - 1
#         with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
#             if decoder_layer == 0:
#                 # first decoder layer doesn't have skip connections
#                 # since it is directly connected to the skip_layer
#                 input = layers[-1]
#             else:
#                 input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
#
#             rectified = tf.nn.relu(input)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
#             output = deconv(rectified, out_channels)
#             output = batchnorm(output)
#
#             if dropout > 0.0:
#                 output = tf.nn.dropout(output, keep_prob=1 - dropout)
#
#             layers.append(output)
#
#     # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
#     with tf.variable_scope("decoder_1"):
#         input = tf.concat([layers[-1], layers[0]], axis=3)
#         rectified = tf.nn.relu(input)
#         output = deconv(rectified, generator_outputs_channels)
#         output = tf.tanh(output)
#         layers.append(output)
#
#     return layers[-1]


def create_res_net(generator_inputs, generator_outputs_channels, n_res_blocks=9, ngf=32):
    layers = []

    encoder(generator_inputs, layers, ngf)

    # 9 residual blocks = r128: [batch, 64, 64, ngf*4] => [batch, 64, 64, ngf*4]
    with tf.variable_scope("resnet"):
        for block in range(n_res_blocks):
            with tf.variable_scope("residual_block_%d" % (block + 1)):
                input = layers[-1]
                output = input
                for layer in range(2):
                    with tf.variable_scope("layer_%d" % (layer + 1)):
                        output = conv(output, ngf * 4, size=3, stride=1)
                        output = batchnorm(output)
                        output = tf.nn.relu(output)
                layers.append(input+output)

    decoder(generator_outputs_channels, layers, ngf)

    return layers[-1]


def create_highway_net(generator_inputs, generator_outputs_channels, n_highway_units=9, ngf=32):
    layers = []

    encoder(generator_inputs, layers, ngf)

    # 9 residual blocks = r128: [batch, 64, 64, ngf*4] => [batch, 64, 64, ngf*4]
    with tf.variable_scope("highwaynet"):
        for block in range(n_highway_units):
            with tf.variable_scope("highway_unit_%d" % (block + 1)):
                input = layers[-1]
                with tf.variable_scope("transform"):
                    output = input
                    for layer in range(2):
                        with tf.variable_scope("layer_%d" % (layer + 1)):
                            output = conv(output, ngf * 4, size=3, stride=1)
                            output = batchnorm(output)
                            output = tf.nn.relu(output)
                with tf.variable_scope("gate"):
                    gate = conv(input, ngf * 4, size=3, stride=1, initializer=tf.constant_initializer(-1.0))
                    output = batchnorm(output)
                    gate = tf.nn.sigmoid(gate)

                layers.append(input*(1.0-gate) + output*gate)

    decoder(generator_outputs_channels, layers, ngf)

    return layers[-1]


def create_dense_net(generator_inputs, generator_outputs_channels, n_dense_blocks=5, n_dense_layers=5, ngf=32):
    layers = []

    encoder(generator_inputs, layers, ngf)

    # n_layers = n_dense_blocks * n_dense_layers
    with tf.variable_scope("densenet"):
        for block in range(n_dense_blocks):
            with tf.variable_scope("dense_block_%d" % (block + 1)):
                nodes = []
                nodes.append(layers[-1])
                for layer in range(n_dense_layers):
                    with tf.variable_scope("dense_layer_%d" % (layer + 1)):
                        input = tf.concat(nodes, 3)
                        output = conv(input, ngf * 4, size=3, stride=1)
                        output = batchnorm(output)
                        output = tf.nn.relu(output)
                        nodes.append(output)
                layers.append(nodes[-1])

    decoder(generator_outputs_channels, layers, ngf)

    return layers[-1]


def encoder(generator_inputs, layers, ngf):
    with tf.variable_scope("encoder"):
        # encoder_1 = c7s1 - 32: [batch, 256, 256, in_channels] => [batch, 256, 256, ngf]
        with tf.variable_scope("conv_1"):
            output = conv(generator_inputs, ngf, size=7, stride=1)
            layers.append(output)

        # encoder_2 = d64: [batch, 256, 256, ngf] => [batch, 128, 128, ngf*2]
        with tf.variable_scope("conv_2"):
            output = conv(layers[-1], ngf * 2, size=3, stride=2)
            layers.append(output)

        # encoder_3 = d128: [batch, 128, 128, ngf*2] => [batch, 64, 64, ngf*4]
        with tf.variable_scope("conv_3"):
            output = conv(layers[-1], ngf * 4, size=3, stride=2)
            layers.append(output)


def decoder(generator_outputs_channels, layers, ngf):
    with tf.variable_scope("encoder"):
        # decoder_3 = u64: [batch, 64, 64, ngf*4] => [batch, 128, 128, ngf*2]
        with tf.variable_scope("deconv_1"):
            input = layers[-1]
            output = deconv(input, ngf * 2)
            output = batchnorm(output)
            rectified = tf.nn.relu(output)
            layers.append(rectified)

        # decoder_2 = u32: [batch, 128, 128, ngf*2] => [batch, 256, 256, ngf]
        with tf.variable_scope("deconv_2"):
            input = layers[-1]
            output = deconv(input, ngf)
            output = batchnorm(output)
            rectified = tf.nn.relu(output)
            layers.append(rectified)

        # decoder_1 = c7s1-3: [batch, 256, 256, ngf] => [batch, 256, 256, generator_output_channels]
        with tf.variable_scope("deconv_3"):
            input = layers[-1]
            output = conv(input, generator_outputs_channels, size=7, stride=1)
            output = tf.tanh(output)
            layers.append(output)


if a.generator == 'unet':
    create_generator = create_u_net
elif a.generator == 'resnet':
    create_generator = create_res_net
elif a.generator == 'highwaynet':
    create_generator = create_highway_net
elif a.generator == 'densenet':
    create_generator = create_dense_net

def create_discriminator(input):
    n_layers = 3
    layers = []

    # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def log_loss(real, fake):
    # minimizing -tf.log(x) will try to get x to 1
    # predict_real => 1
    # predict_fake => 0
    if a.model == 'CycleGAN':   # unpaired images in loss
        with tf.name_scope("log_loss_unpaired_images"):
            result = tf.reduce_mean(-tf.log(real + EPS)) + tf.reduce_mean(-tf.log(1 - fake + EPS))
    else:   # paired images in loss
        with tf.name_scope("log_loss_paired_images"):
            result = tf.reduce_mean(-(tf.log(real + EPS) + tf.log(1 - fake + EPS)))
    return result


def square_loss(real, fake):
    # minimizing tf.square(1 - x) will try to get x to 1
    # predict_real => 1
    # predict_fake => 0
    if a.model == 'CycleGAN':  # unpaired images in loss
        result = tf.reduce_mean(tf.square(real - 1)) + tf.reduce_mean(tf.square(fake))
    else:   # paired images in loss
        result = tf.reduce_mean(tf.square(real - 1) + tf.square(fake))
    return result


if a.loss == "log":
    loss = log_loss
elif a.loss == "square":
    loss = square_loss


def GAN_loss(discrim_loss, fake, real):
    if a.gen_loss == 'fake':
        if a.loss == "log":
            # original implementation: negative log loss on fake only
            # predict_fake => 1
            result = tf.reduce_mean(-tf.log(fake + EPS))
        elif a.loss == "square":
            result = tf.reduce_mean(tf.square(fake - 1))
    elif a.gen_loss == 'negative':
        # maximising discriminator loss (on real over fake)
        result = -discrim_loss
    elif a.gen_loss == 'contra':
        # minimising discriminator loss on fake over real
        result = loss(fake, real)
    return result


def classic_loss(outputs, targets, target_type):
    if target_type == "image":  # Absolute value loss / L1 loss
        gen_loss_classic = tf.reduce_mean(tf.abs(targets - outputs))
    elif target_type == "label":  # Cross entropy loss
        # [-1,+1] ==> [0, 1] for labels
        gen_loss_classic = tf.reduce_mean(tf.losses.softmax_cross_entropy(targets/2+0.5, outputs/2+0.5))
    else:
        raise ValueError("Unknown target type", target_type)
    return gen_loss_classic


def create_pix2pix_model(inputs, targets,
                         generator_name="generator", discriminator_name="discriminator", target_type=a.X_type):
    with tf.variable_scope(generator_name) as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    def create_discriminator_for_image_pairs(discrim_inputs, discrim_targets):
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
        return create_discriminator(input)

    with tf.name_scope(discriminator_name+"_on_real"):
        with tf.variable_scope(discriminator_name):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator_for_image_pairs(inputs, targets)

    with tf.name_scope(discriminator_name+"_on_fake"):
        with tf.variable_scope(discriminator_name, reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator_for_image_pairs(inputs, outputs)

    with tf.name_scope("loss_"+discriminator_name):
        discrim_loss = loss(predict_real, predict_fake)

    with tf.name_scope("loss_"+generator_name):
        gen_loss_GAN = GAN_loss (discrim_loss, predict_fake, predict_real)
        gen_loss_classic = classic_loss(outputs, targets, target_type)
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_classic * a.classic_weight

    with tf.name_scope("train_"+discriminator_name):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith(discriminator_name)]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("train_"+generator_name):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith(generator_name)]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_classic])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_classic=ema.average(gen_loss_classic),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def create_pix2pix2_model(X, Y):
    forward_model = create_pix2pix_model(X, Y,
                                         generator_name="G", discriminator_name="D_Y", target_type=a.Y_type)
    reverse_model = create_pix2pix_model(Y, X,
                                         generator_name="F", discriminator_name="D_X", target_type=a.X_type)
    return Pix2Pix2Model(
        predict_real_X=reverse_model.predict_real,
        predict_fake_X=reverse_model.predict_fake,
        predict_real_Y=forward_model.predict_real,
        predict_fake_Y=forward_model.predict_fake,
        discrim_X_loss=reverse_model.discrim_loss,
        discrim_Y_loss=forward_model.discrim_loss,
        discrim_X_grads_and_vars=reverse_model.discrim_grads_and_vars,
        discrim_Y_grads_and_vars=reverse_model.discrim_grads_and_vars,
        gen_G_loss_GAN=forward_model.gen_loss_GAN,
        gen_F_loss_GAN=reverse_model.gen_loss_GAN,
        gen_G_loss_classic=forward_model.gen_loss_classic,
        gen_F_loss_classic=reverse_model.gen_loss_classic,
        gen_G_grads_and_vars=forward_model.gen_grads_and_vars,
        gen_F_grads_and_vars=reverse_model.gen_grads_and_vars,
        outputs=forward_model.outputs,
        reverse_outputs=reverse_model.outputs,
        train=tf.group(forward_model.train, reverse_model.train)
    )


def create_CycleGAN_model(X, Y):
    # create two generators G and F, one for forward and one for backward translation, each having two copies,
    # one for real images and one for fake images which share the same underlying variables
    with tf.name_scope("G_on_real"):
        with tf.variable_scope("G") as scope:
            Y_channels = int(Y.get_shape()[-1])
            fake_Y = create_generator(X, Y_channels)

    with tf.name_scope("F_on_real"):
        with tf.variable_scope("F") as scope:
            X_channels = int(X.get_shape()[-1])
            fake_X = create_generator(Y, X_channels)

    with tf.name_scope("G_on_fake"):
        with tf.variable_scope("G", reuse=True) as scope:
            Y_channels = int(Y.get_shape()[-1])
            fake_Y_from_fake_X = create_generator(fake_X, Y_channels)

    with tf.name_scope("F_on_fake"):
        with tf.variable_scope("F", reuse=True) as scope:
            X_channels = int(X.get_shape()[-1])
            fake_X_from_fake_Y = create_generator(fake_Y, X_channels)

    # create two discriminators D_X and D_Y, each having two copies,
    # one for real images and one for fake image which share the same underlying variables
    with tf.name_scope("D_X_on_real"):
        with tf.variable_scope("D_X"):
            # [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_X = create_discriminator(X)

    with tf.name_scope("D_X_on_fake"):
        with tf.variable_scope("D_X", reuse=True):
            # [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_X = create_discriminator(fake_X)

    with tf.name_scope("D_Y_on_real"):
        with tf.variable_scope("D_Y"):
            # [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_Y = create_discriminator(Y)

    with tf.name_scope("D_Y_on_fake"):
        with tf.variable_scope("D_Y", reuse=True):
            # [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_Y = create_discriminator(fake_Y)

    # define loss for D_X and D_Y
    with tf.name_scope("loss_D_X"):
        discrim_X_loss = loss(predict_real_X, predict_fake_X)

    with tf.name_scope("loss_D_Y"):
        discrim_Y_loss = loss(predict_real_Y, predict_fake_Y)

    # define cycle_consistency loss, one for foward one for backward
    with tf.name_scope("loss_cycle_consistency"):
        forward_loss_classic = classic_loss(fake_X_from_fake_Y, X, a.X_type)
        backward_loss_classic = classic_loss(fake_Y_from_fake_X, Y, a.Y_type)
        cycle_consistency_loss_classic = forward_loss_classic + backward_loss_classic

    # define loss for G and F
    with tf.name_scope("loss_G"):
        # predict_fake => 1
        # abs() => 0
        gen_G_loss_GAN = GAN_loss(discrim_Y_loss, predict_fake_Y, predict_real_Y)
        gen_G_loss = gen_G_loss_GAN * a.gan_weight + cycle_consistency_loss_classic * a.classic_weight

    with tf.name_scope("loss_F"):
        # predict_fake => 1
        # abs() => 0
        # gen_F_loss_GAN = tf.reduce_mean(-tf.log(predict_fake_X + EPS))
        # gen_F_loss_GAN = -discrim_X_loss
        gen_F_loss_GAN = GAN_loss(discrim_X_loss, predict_fake_X, predict_real_X)
        gen_F_loss = gen_F_loss_GAN * a.gan_weight + cycle_consistency_loss_classic * a.classic_weight

    # train discriminators
    def train_discriminator(prefix, discrim_loss):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith(prefix)]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        return discrim_grads_and_vars, discrim_train

    with tf.name_scope("train_D_Y"):
        discrim_Y_grads_and_vars, discrim_Y_train = train_discriminator("D_Y", discrim_Y_loss)

    with tf.name_scope("train_D_X"):
        discrim_X_grads_and_vars, discrim_X_train = train_discriminator("D_X", discrim_X_loss)

    # train generators
    def train_generator(prefix, gen_loss):
        if a.untouch == "nothing":
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith(prefix)]
        elif a.untouch == "core":
            gen_tvars = [var for var in tf.trainable_variables()
                         if var.name.startswith(prefix) and not var.name.startswith(prefix+"/"+a.generator)]
            print("Exclude %s %s/%s from training" % (a.untouch, prefix, a.generator))
        gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
        return gen_grads_and_vars, gen_train

    with tf.name_scope("train_G"):
        with tf.control_dependencies([discrim_Y_train]):
            gen_G_grads_and_vars, gen_G_train = train_generator("G", gen_G_loss)

    with tf.name_scope("train_F"):
        with tf.control_dependencies([discrim_X_train]):
            gen_F_grads_and_vars, gen_F_train = train_generator("F", gen_F_loss)

    # other variables
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_X_loss, discrim_Y_loss,
                               gen_G_loss_GAN, gen_F_loss_GAN,
                               forward_loss_classic, backward_loss_classic,
                               cycle_consistency_loss_classic])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return CycleGANModel(
        predict_real_X=predict_real_X,
        predict_fake_X=predict_fake_X,
        predict_real_Y=predict_real_Y,
        predict_fake_Y=predict_fake_Y,
        discrim_X_loss=ema.average(discrim_X_loss),
        discrim_Y_loss=ema.average(discrim_Y_loss),
        discrim_X_grads_and_vars=discrim_X_grads_and_vars,
        discrim_Y_grads_and_vars=discrim_Y_grads_and_vars,
        gen_G_loss_GAN=ema.average(gen_G_loss_GAN),
        gen_F_loss_GAN=ema.average(gen_F_loss_GAN),
        forward_cycle_loss_classic=ema.average(forward_loss_classic),
        backward_cycle_loss_classic=ema.average(backward_loss_classic),
        gen_G_grads_and_vars=gen_G_grads_and_vars,
        gen_F_grads_and_vars=gen_F_grads_and_vars,
        outputs=fake_Y,
        reverse_outputs=fake_X,
        train=tf.group(update_losses, incr_global_step, gen_G_train, gen_F_train),
        cycle_consistency_loss_classic=ema.average(cycle_consistency_loss_classic),
    )


if a.model =="pix2pix":
    image_kinds = ["inputs", "outputs", "targets"]
else:
    image_kinds = ["inputs", "reverse_outputs", "outputs", "targets"]


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["input_paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        if not a.model == 'pix2pix':
            target_path =  fetches["target_paths"][i]
            name2, _ = os.path.splitext(os.path.basename(target_path.decode("utf8")))
            fileset["name2"] = name2
        for kind in image_kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        if a.model == 'pix2pix':
            index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
        else:
            index.write("<th>name</th><th>input</th><th>reverse_output</th><th>output</th><th>target</th><th>name</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in image_kinds:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        if not a.model == 'pix2pix':
            index.write("<td>%s</td>" % fileset["name2"])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()

    # inputs and targets are [batch_size, height, width, channels]
    if a.model == 'pix2pix':
        model = create_pix2pix_model(examples.inputs, examples.targets)
    elif a.model == 'pix2pix2':
        model = create_pix2pix2_model(examples.inputs, examples.targets)
    elif a.model == 'CycleGAN':
        model = create_CycleGAN_model(examples.inputs, examples.targets)

    # encoding images for saving
    with tf.name_scope("encode_images"):
        display_fetches = {}
        for name, value in examples._asdict().iteritems():
            if "path" in name:
                display_fetches[name] = value
            elif tf.is_numeric_tensor(value):
                display_fetches[name] = tf.map_fn(tf.image.encode_png, deprocess(value), dtype=tf.string, name=name+"_pngs")
        for name, value in model._asdict().iteritems():
            if tf.is_numeric_tensor(value) and "predict_" not in name:
                display_fetches[name] = tf.map_fn(tf.image.encode_png, deprocess(value), dtype=tf.string, name=name+"_pngs")

    # progress report for all losses
    with tf.name_scope("progress_summary"):
        progress_fetches = {}
        for name, value in model._asdict().iteritems():
            if not tf.is_numeric_tensor(value) and "grads_and_vars" not in name and not name == "train":
                progress_fetches[name] = value

    # summaries for model: images, scalars, histograms
    for name, value in examples._asdict().iteritems():
        if tf.is_numeric_tensor(value):
            with tf.name_scope(name + "_summary"):
                tf.summary.image(name, deprocess(value))
    for name, value in model._asdict().iteritems():
        if tf.is_numeric_tensor(value):
            with tf.name_scope(name + "_summary"):
                if "predict_" in name:    # discriminators produce values in [0, 1]
                    tf.summary.image(name, tf.image.convert_image_dtype(value, dtype=tf.uint8))
                else:   # generators produce values in [-1, 1]
                    tf.summary.image(name, deprocess(value))
        elif "grads_and_vars" in name:
            for grad, var in value:
                tf.summary.histogram(var.op.name + "/gradients", grad)
        elif not name == "train":
            tf.summary.scalar(name, value)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    if a.restore=="generators":
        print("restore only generators")
        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G') \
                            + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='F')
        restore_saver = tf.train.Saver(restore_variables)
    else:
        restore_saver = saver

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading "+a.restore+" from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["progress"] = progress_fetches

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    for name, value in results["progress"].iteritems():
                        print (name, value)

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
