
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_rad2deg_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(np.pi)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    x = tf.constant(np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = tf.constant(np.array([[0, np.pi/4], [np.pi/2, np.pi]]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    x = tf.constant(np.array([[[0, np.pi/6], [np.pi/3, np.pi/2]], [[np.pi, 7*np.pi/6], [4*np.pi/3, 3*np.pi/2]]]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    x = tf.constant(np.array([-np.pi/2, -np.pi, -3*np.pi/2]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with mixed positive and negative values
    x = tf.constant(np.array([-np.pi, -np.pi/4, 0, np.pi/4, np.pi]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values
    x = tf.constant(np.array([10*np.pi, 20*np.pi]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values
    x = tf.constant(np.array([np.pi/100, np.pi/1000]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero value
    x = tf.constant(np.array([0.0]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: High dimensional array
    x = tf.constant(np.random.rand(2, 2, 2, 2) * np.pi)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.rad2deg"] = tf_experimental_numpy_rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.rad2deg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.rad2deg'.")

check_valid('tf.experimental.numpy.rad2deg', generated_inputs['tf.experimental.numpy.rad2deg'], lib="tf", suffix=0)
