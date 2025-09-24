
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_cos_inputs():
    list_of_inputs = []

    # Input 1: Scalar input
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    x = tf.constant([0.0, np.pi/2, np.pi])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    x = tf.constant([[0.0, np.pi/4], [np.pi/2, np.pi]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    x = tf.constant([[[0.0, np.pi/2], [np.pi, 3*np.pi/2]], [[2*np.pi, 5*np.pi/2], [3*np.pi, 7*np.pi/2]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    x = tf.constant([-np.pi, -np.pi/2, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    x = tf.constant([10*np.pi, 20*np.pi])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small values
    x = tf.constant([0.0001, -0.0001])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: tf.float16
    x = tf.constant([0.0, np.pi/2], dtype=tf.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tf.float64
    x = tf.constant([0.0, np.pi/2], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Mixed positive and negative values
    x = tf.constant([-np.pi/4, np.pi/4, -3*np.pi/4, 3*np.pi/4])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        input_dict['x'] = input_dict['x'].numpy()
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.cos"] = tf_experimental_numpy_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cos'.")

check_valid('tf.experimental.numpy.cos', generated_inputs['tf.experimental.numpy.cos'], lib="tf", suffix=0)
