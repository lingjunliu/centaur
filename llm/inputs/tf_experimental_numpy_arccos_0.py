
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arccos_inputs():
    list_of_inputs = []

    # Input 1: Valid input tensor
    x = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid input tensor with float64 dtype
    x = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid input tensor with negative values
    x = tf.constant([-0.2, -0.4, -0.6, -0.8, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid input tensor with positive values
    x = tf.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid input tensor with zeros
    x = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid input tensor, multidimensional
    x = tf.constant([[-1.0, -0.5, 0.0], [0.5, 1.0, 0.2]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Valid input tensor, all values within valid range
    x = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid input tensor with a mix of positive and negative values
    x = tf.constant([-0.7, 0.3, -0.1, 0.9, -0.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid input tensor - all ones and negative ones
    x = tf.constant([-1.0, -1.0, 1.0, 1.0, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid input - numpy array converted to tensor
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.arccos"] = tf_experimental_numpy_arccos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arccos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arccos'.")

check_valid('tf.experimental.numpy.arccos', generated_inputs['tf.experimental.numpy.arccos'], lib="tf", suffix=0)
