
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_extract_jpeg_shape_inputs():
    list_of_inputs = []

    # Input 1: Basic valid JPEG data
    jpeg_data_1 = tf.io.encode_jpeg(tf.zeros([100, 100, 3], dtype=tf.uint8)).numpy()
    input_dict_1 = {"contents": jpeg_data_1, "output_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different output type (int64)
    jpeg_data_2 = tf.io.encode_jpeg(tf.zeros([50, 50, 3], dtype=tf.uint8)).numpy()
    input_dict_2 = {"contents": jpeg_data_2, "output_type": tf.int64, "name": "jpeg_shape_2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: JPEG data with a comment
    jpeg_data_3 = tf.io.encode_jpeg(tf.zeros([200, 300, 3], dtype=tf.uint8), quality=50).numpy()
    input_dict_3 = {"contents": jpeg_data_3, "output_type": tf.int32, "name": "jpeg_shape_3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Smaller image
    jpeg_data_4 = tf.io.encode_jpeg(tf.zeros([10, 10, 3], dtype=tf.uint8)).numpy()
    input_dict_4 = {"contents": jpeg_data_4, "output_type": tf.int64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Grayscale image
    jpeg_data_5 = tf.io.encode_jpeg(tf.zeros([100, 100, 1], dtype=tf.uint8)).numpy()
    input_dict_5 = {"contents": jpeg_data_5, "output_type": tf.int32, "name": "jpeg_shape_5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large image
    jpeg_data_6 = tf.io.encode_jpeg(tf.zeros([1000, 1000, 3], dtype=tf.uint8)).numpy()
    input_dict_6 = {"contents": jpeg_data_6, "output_type": tf.int64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Image with quality setting
    jpeg_data_7 = tf.io.encode_jpeg(tf.zeros([64, 64, 3], dtype=tf.uint8), quality=95).numpy()
    input_dict_7 = {"contents": jpeg_data_7, "output_type": tf.int32, "name": "jpeg_shape_7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another slightly different image size
    jpeg_data_8 = tf.io.encode_jpeg(tf.zeros([128, 256, 3], dtype=tf.uint8)).numpy()
    input_dict_8 = {"contents": jpeg_data_8, "output_type": tf.int64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Different quality setting
    jpeg_data_9 = tf.io.encode_jpeg(tf.zeros([32, 32, 3], dtype=tf.uint8), quality=10).numpy()
    input_dict_9 = {"contents": jpeg_data_9, "output_type": tf.int32, "name": "jpeg_shape_9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Image with no name specified
    jpeg_data_10 = tf.io.encode_jpeg(tf.zeros([256, 128, 3], dtype=tf.uint8)).numpy()
    input_dict_10 = {"contents": jpeg_data_10, "output_type": tf.int64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.extract_jpeg_shape"] = tf_io_extract_jpeg_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.extract_jpeg_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.extract_jpeg_shape'.")

check_valid('tf.io.extract_jpeg_shape', generated_inputs['tf.io.extract_jpeg_shape'], lib="tf", suffix=0)
