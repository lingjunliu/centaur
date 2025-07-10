
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tanh_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    x = np.array(0.0)
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    x = np.array([-1.0, 0.0, 1.0])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = np.array([[-2.0, -1.0], [0.0, 1.0]])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values
    x = np.array([10.0, -10.0])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small values
    x = np.array([0.001, -0.001])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed values
    x = np.array([-5.0, 0.0, 5.0, -0.5, 0.5])
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros
    x = np.zeros((2, 2))
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All ones
    x = np.ones((2, 2))
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different data type (float64)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tanh"] = tf_experimental_numpy_tanh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tanh'.")

check_valid('tf.experimental.numpy.tanh', generated_inputs['tf.experimental.numpy.tanh'], lib="tf", suffix=0)
