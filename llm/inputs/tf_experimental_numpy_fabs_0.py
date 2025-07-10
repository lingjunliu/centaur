
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_fabs_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic negative values
    x = tf.constant(np.array([-1.0, -2.0, -3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative values
    x = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values
    x = tf.constant(np.array([-1.0, 0.0, -3.0, 0.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = tf.constant(np.array([[-1.0, 2.0], [-3.0, 4.0]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    x = tf.constant(np.array([[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [-7.0, 8.0]]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer values
    x = tf.constant(np.array([-1, 2, -3, 4]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 values
    x = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger values
    x = tf.constant(np.array([-100.0, 200.0, -300.0, 400.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with a mix of 0 and small floats
    x = tf.constant(np.array([-0.1, 0.0, 0.2, -0.001]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.fabs"] = tf_experimental_numpy_fabs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.fabs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.fabs'.")

check_valid('tf.experimental.numpy.fabs', generated_inputs['tf.experimental.numpy.fabs'], lib="tf", suffix=0)
