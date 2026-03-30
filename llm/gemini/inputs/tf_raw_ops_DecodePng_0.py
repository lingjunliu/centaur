
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def _encode_png(image):
    """Helper function to encode a TensorFlow tensor as a PNG image."""
    png_encoded = tf.io.encode_png(tf.cast(image, tf.uint8)).numpy()
    return png_encoded.tobytes().decode('latin-1')


def tf_raw_ops_DecodePng_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 0
    dtype = tf.uint8
    name = None
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[0, 255, 0]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 1
    dtype = tf.uint8
    name = "decode_green"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[0, 0, 255]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 3
    dtype = tf.uint8
    name = "decode_blue"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 4
    dtype = tf.uint8
    name = "decode_white"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 0
    dtype = tf.uint8
    name = "decode_gray"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 0
    dtype = tf.uint16
    name = "decode_gray_uint16"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 1
    dtype = tf.uint16
    name = "decode_gray_uint16_single_channel"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 3
    dtype = tf.uint16
    name = "decode_gray_uint16_three_channels"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Larger image
    image = np.array([[[100, 150, 200] for _ in range(10)] for _ in range(10)], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 0
    dtype = tf.uint8
    name = "decode_gray_large_image"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - RGBA
    image = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
    contents = _encode_png(image)
    channels = 4
    dtype = tf.uint16
    name = "decode_rgba_uint16"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodePng"] = tf_raw_ops_DecodePng_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePng'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DecodePng', generated_inputs['tf.raw_ops.DecodePng'], lib="tf", suffix=0)
