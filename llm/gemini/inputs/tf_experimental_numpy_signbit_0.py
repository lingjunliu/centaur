
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_signbit_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(-5.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    x = tf.constant([-1.0, 0.0, 1.0, -2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    x = tf.constant([[[1.0, -2.0], [-3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive values only
    x = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values only
    x = tf.constant([-1.0, -2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive, negative, and zero
    x = tf.constant([-1.0, 0.0, 2.0, -3.0, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    x = tf.constant([-1e9, 1e9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small values
    x = tf.constant([-1e-9, 1e-9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tf.float64 tensor
    x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Convert tf.Tensor to numpy array
    for i in range(len(list_of_inputs)):
        list_of_inputs[i]['x'] = list_of_inputs[i]['x'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.signbit"] = tf_experimental_numpy_signbit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.signbit'.")

check_valid('tf.experimental.numpy.signbit', generated_inputs['tf.experimental.numpy.signbit'], lib="tf", suffix=0)
