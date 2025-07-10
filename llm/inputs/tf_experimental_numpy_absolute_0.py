
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_absolute_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    x = tf.constant(-5.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor with positive and negative values
    x = tf.constant([-1, 2, -3, 4, -5], dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor with float values
    x = tf.constant([[-1.5, 2.5], [-3.5, 4.5]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with all zero values
    x = tf.zeros((3, 3), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with all positive values
    x = tf.ones((2, 2), dtype=tf.int32) * 5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with mixed data types (int and float)
    x = tf.constant([-1, 2.5, -3, 4.5], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large tensor with random values
    x = tf.random.normal((100, 100), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with negative infinity
    x = tf.constant([-float('inf'), -1.0, 0.0, 1.0, float('inf')], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with NaN values
    x = tf.constant([-float('nan'), -1.0, 0.0, 1.0, float('nan')], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.absolute"] = tf_experimental_numpy_absolute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.absolute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.absolute'.")

check_valid('tf.experimental.numpy.absolute', generated_inputs['tf.experimental.numpy.absolute'], lib="tf", suffix=0)
