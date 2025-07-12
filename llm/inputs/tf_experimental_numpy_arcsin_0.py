
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arcsin_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    x = np.array([[0.2, 0.4], [-0.6, -0.8]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [-0.3, -0.4]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with values close to 1 and -1
    x = np.array([0.99, -0.99, 0.999, -0.999])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with a mix of positive and negative values near 0
    x = np.array([0.01, -0.01, 0.001, -0.001])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with all zeros
    x = np.zeros((2, 3))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with all ones (should result in nan)
    x = np.array([0.5, -0.5, 0.9, -0.9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large array
    x = np.linspace(-0.9, 0.9, 100)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with dtype float64
    x = np.array([-0.7, 0.3, -0.1, 0.8], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with dtype float32
    x = np.array([0.6, -0.2, 0.4, -0.9], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.arcsin"] = tf_experimental_numpy_arcsin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arcsin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arcsin'.")

check_valid('tf.experimental.numpy.arcsin', generated_inputs['tf.experimental.numpy.arcsin'], lib="tf", suffix=0)
