
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tan_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    x = tf.constant([0.0, np.pi/4, np.pi/2, np.pi])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = tf.constant([[0.0, np.pi/4], [np.pi/2, np.pi]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = tf.constant([-np.pi/4, -np.pi/2, -np.pi])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values
    x = tf.constant([10.0, 20.0, 30.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small values
    x = tf.constant([0.001, 0.0001, 0.00001])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive and negative
    x = tf.constant([-1.0, 0.0, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with zeros
    x = tf.constant([0.0, 1.0, 0.0, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array
    x = tf.constant([[[0.0, np.pi/4], [np.pi/2, np.pi]], [[-np.pi/4, 0.0], [np.pi, np.pi/2]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tf.float64
    x = tf.constant(np.array([0.0, np.pi/4, np.pi/2, np.pi], dtype=np.float64))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tan"] = tf_experimental_numpy_tan_inputs()

for k, v in generated_inputs.items():
    for i, input_dict in enumerate(v):
        for key, value in input_dict.items():
            if isinstance(value, tf.Tensor):
                input_dict[key] = value.numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tan'.")

check_valid('tf.experimental.numpy.tan', generated_inputs['tf.experimental.numpy.tan'], lib="tf", suffix=0)
