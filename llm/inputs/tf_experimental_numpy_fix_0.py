
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_fix_inputs():
    list_of_inputs = []

    # Input 1: Scalar integer
    x = tf.constant(5, dtype=tf.int32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float
    x = tf.constant(5.7, dtype=tf.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar negative float
    x = tf.constant(-2.3, dtype=tf.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor of integers
    x = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor of floats
    x = tf.constant([1.1, 2.5, 3.9, 4.0, 5.7], dtype=tf.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor of integers
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.int32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor of floats
    x = tf.constant([[1.1, 2.5], [3.9, 4.0]], dtype=tf.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor of integers
    x = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with negative values
    x = tf.constant([-1.5, 2.7, -3.2, 4.9], dtype=tf.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    x = tf.constant([1000.1, -2000.7, 3000.2], dtype=tf.float32).numpy()
    input_dict = {"x": x}
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
