
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Array with negative integers
    x = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array with mixed positive and negative integers
    x = np.array([-1, 2, -3, 4, -5], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with zeros
    x = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger integers
    x = np.array([255, 65535, 2147483647], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Array with different data type (int16)
    x = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with different data type (int64)
    x = np.array([-1, -2, -3, -4, -5], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: array with a single element
    x = np.array([10], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.bitwise_not"] = tf_experimental_numpy_bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.bitwise_not'.")

check_valid('tf.experimental.numpy.bitwise_not', generated_inputs['tf.experimental.numpy.bitwise_not'], lib="tf", suffix=0)
