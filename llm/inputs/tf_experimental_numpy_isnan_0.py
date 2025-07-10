
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isnan_inputs():
    list_of_inputs = []

    # Input 1: Basic NaN values
    x = tf.constant([np.nan, 1.0, np.nan, 2.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No NaN values
    x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Only NaN values
    x = tf.constant([np.nan, np.nan, np.nan], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer type (no NaN representation)
    x = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with NaN
    x = tf.constant([[np.nan, 1.0], [2.0, np.nan]], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with NaN and regular values
    x = tf.constant([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, np.nan]]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex number with NaN in real part
    x = tf.constant([complex(np.nan, 1.0), complex(2.0, 3.0)], dtype=tf.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex number with NaN in imaginary part
    x = tf.constant([complex(1.0, np.nan), complex(2.0, 3.0)], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with mixed NaN and inf values
    x = tf.constant([np.nan, np.inf, 1.0, -np.inf], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor
    x = tf.constant(np.array([]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([np.nan]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.isnan"] = tf_experimental_numpy_isnan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isnan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isnan'.")

check_valid('tf.experimental.numpy.isnan', generated_inputs['tf.experimental.numpy.isnan'], lib="tf", suffix=0)
