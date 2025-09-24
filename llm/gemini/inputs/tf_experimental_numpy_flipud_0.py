
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_flipud_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array
    m = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    m = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    m = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty array (0x2)
    m = tf.constant([], dtype=tf.int32, shape=(0, 2))
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with different data type (float)
    m = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with negative values
    m = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Array with zeros
    m = tf.constant([[0, 0], [0, 0]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array
    m = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with a single element
    m = tf.constant([[5]], dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex type. Removing this since complex64 might cause issues in size calculation later.
    # m = tf.constant([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=tf.complex64)
    # input_dict = {"m": m}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.flipud"] = tf_experimental_numpy_flipud_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.flipud'.")

check_valid('tf.experimental.numpy.flipud', generated_inputs['tf.experimental.numpy.flipud'], lib="tf", suffix=0)
