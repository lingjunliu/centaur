
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_cbrt_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(-8.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    x = tf.constant([1.0, 8.0, 27.0, 64.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    x = tf.constant([[1.0, -8.0], [27.0, -64.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    x = tf.constant([[[1.0, 8.0], [27.0, 64.0]], [[-1.0, -8.0], [-27.0, -64.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: tensor with zeros
    x = tf.constant([0.0, 0.0, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tensor with large values
    x = tf.constant([1000.0, -1000.0, 1000000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tensor with fractional values
    x = tf.constant([0.125, -0.125, 0.008])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: tensor with different dtypes (float32)
    x = tf.constant([1.0, 8.0, 27.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tensor with different dtypes (float64)
    x = tf.constant([1.0, 8.0, 27.0], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tensor with negative and positive values
    x = tf.constant([-1.0, 8.0, -27.0, 64.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.cbrt"] = tf_experimental_numpy_cbrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.cbrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cbrt'.")

check_valid('tf.experimental.numpy.cbrt', generated_inputs['tf.experimental.numpy.cbrt'], lib="tf", suffix=0)
