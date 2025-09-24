
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_heaviside_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive and negative values
    x1 = np.array([-1.5, 0, 2.0])
    x2 = np.array([0.5, 1.0, 1.5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x1 is all zeros
    x1 = np.array([0, 0, 0])
    x2 = np.array([0.5, 1.0, 1.5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x2 is all zeros
    x1 = np.array([-1.5, 0, 2.0])
    x2 = np.array([0, 0, 0])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both x1 and x2 are all zeros
    x1 = np.array([0, 0, 0])
    x2 = np.array([0, 0, 0])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimension array
    x1 = np.array([[1, 0, -1], [-2, 2, 0]])
    x2 = np.array([[0.5, 1, 1.5], [1, 0, 2]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shaped arrays (broadcastable)
    x1 = np.array([1, 0, -1])
    x2 = np.array([0.5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 with large positive and negative numbers
    x1 = np.array([-1e9, 0, 1e9])
    x2 = np.array([0.5, 1.0, 1.5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x2 with large values
    x1 = np.array([-1.5, 0, 2.0])
    x2 = np.array([0.5, 1e9, 1.5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64
    x1 = np.array([-1.5, 0, 2.0], dtype=np.float64)
    x2 = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: broadcasting with scalar x2
    x1 = np.array([[1, 2, 3], [4, 5, 6]])
    x2 = np.array(1.0)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.heaviside"] = tf_experimental_numpy_heaviside_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.heaviside' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.heaviside'.")

check_valid('tf.experimental.numpy.heaviside', generated_inputs['tf.experimental.numpy.heaviside'], lib="tf", suffix=0)
