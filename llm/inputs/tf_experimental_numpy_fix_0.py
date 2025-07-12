
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_fix_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    x = np.array(2.7, dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative float
    x = np.array(-3.14, dtype=np.float64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of floats
    x = np.array([1.2, 2.8, -3.5, 4.9], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of floats
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of floats
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of integers
    x = np.array([1, 2, -3, 4], dtype=np.int32)
    input_dict = {"x": tf.cast(tf.constant(x), dtype=tf.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of integers
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"x": tf.cast(tf.constant(x), dtype=tf.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large float
    x = np.array(123456789.987, dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small float
    x = np.array(0.0000001, dtype=np.float64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with mixed positive and negative floats
    x = np.array([-2.3, 1.7, -0.5, 4.2, -5.8], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.fix"] = tf_experimental_numpy_fix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.fix'.")

check_valid('tf.experimental.numpy.fix', generated_inputs['tf.experimental.numpy.fix'], lib="tf", suffix=0)
