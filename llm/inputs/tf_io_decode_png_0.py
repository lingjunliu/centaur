
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_png_inputs():
    list_of_inputs = []

    # Input 1: Basic valid PNG
    png_data = tf.io.encode_png(tf.constant([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 0, "dtype": tf.uint8, "name": "decode_basic"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale PNG
    png_data = tf.io.encode_png(tf.constant([[[100], [200]], [[50], [150]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 1, "dtype": tf.uint8, "name": "decode_grayscale"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB PNG, request RGBA
    png_data = tf.io.encode_png(tf.constant([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 4, "dtype": tf.uint8, "name": "decode_rgba"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: RGB PNG, request Grayscale
    png_data = tf.io.encode_png(tf.constant([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 1, "dtype": tf.uint8, "name": "decode_to_grayscale"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: RGB PNG, keep RGB
    png_data = tf.io.encode_png(tf.constant([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 3, "dtype": tf.uint8, "name": "decode_rgb"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Simple Grayscale PNG
    png_data = tf.io.encode_png(tf.constant([[[100], [200]], [[50], [150]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 0, "dtype": tf.uint8, "name": "decode_gray"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: PNG with uint16
    image_data = np.array([[[1000, 2000, 3000], [4000, 5000, 6000]], [[7000, 8000, 9000], [10000, 11000, 12000]]], dtype=np.uint16)
    png_data = tf.io.encode_png(tf.constant(image_data)).numpy()
    input_dict = {"contents": png_data, "channels": 0, "dtype": tf.uint16, "name": "decode_uint16"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: PNG with uint16, request channels = 3
    image_data = np.array([[[1000, 2000, 3000], [4000, 5000, 6000]], [[7000, 8000, 9000], [10000, 11000, 12000]]], dtype=np.uint16)
    png_data = tf.io.encode_png(tf.constant(image_data)).numpy()
    input_dict = {"contents": png_data, "channels": 3, "dtype": tf.uint16, "name": "decode_uint16_channels3"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single pixel image
    png_data = tf.io.encode_png(tf.constant([[[255, 0, 0]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 0, "dtype": tf.uint8, "name": "decode_single_pixel"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different sized image
    png_data = tf.io.encode_png(tf.constant([[[128, 64, 32], [32, 64, 128], [64, 128, 32]], [[32, 128, 64], [128, 32, 64], [64, 32, 128]]], dtype=tf.uint8)).numpy()
    input_dict = {"contents": png_data, "channels": 0, "dtype": tf.uint8, "name": "decode_diff_size"}
    input_dict['contents'] = np.array(input_dict['contents'])
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_png"] = tf_io_decode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_png' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_png'.")

check_valid('tf.io.decode_png', generated_inputs['tf.io.decode_png'], lib="tf", suffix=0)
