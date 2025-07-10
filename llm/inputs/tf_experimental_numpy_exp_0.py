
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_exp_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, positive values
    x = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, negative values
    x = tf.constant([-1.0, -2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, mixed values
    x = tf.constant([-1.0, 0.0, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, positive values
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, negative values
    x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, mixed values
    x = tf.constant([[-1.0, 0.0], [1.0, 2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    x = tf.constant([10.0, 20.0, 30.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Float64 array
    x = tf.constant(np.array([1.0, 2.0], dtype=np.float64))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.exp"] = tf_experimental_numpy_exp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.exp'.")

check_valid('tf.experimental.numpy.exp', generated_inputs['tf.experimental.numpy.exp'], lib="tf", suffix=0)
