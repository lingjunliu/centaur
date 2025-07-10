
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_fliplr_inputs():
    list_of_inputs = []

    # Input 1: 2D array
    m = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with float values
    m = tf.constant(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]), dtype=tf.float32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with different shape
    m = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    m = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with single row
    m = tf.constant(np.array([[1, 2, 3, 4]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with single column
    m = tf.constant(np.array([[1], [2], [3], [4]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative values
    m = tf.constant(np.array([[-1, 2, -3], [4, -5, 6]]), dtype=tf.int32)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with boolean values
    m = tf.constant(np.array([[True, False, True], [False, True, False]]), dtype=tf.bool)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with different dtype
    m = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int64)
    input_dict = {"m": m}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.fliplr"] = tf_experimental_numpy_fliplr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.fliplr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.fliplr'.")

check_valid('tf.experimental.numpy.fliplr', generated_inputs['tf.experimental.numpy.fliplr'], lib="tf", suffix=0)
