
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_float_power_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive numbers
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative base with integer exponent
    x1 = np.array([-2.0, -3.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero base
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Fractional exponents
    x1 = np.array([4.0, 9.0], dtype=np.float32)
    x2 = np.array([0.5, 0.5], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional arrays
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[0.5, 1.0], [2.0, 0.5]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed positive and negative exponents
    x1 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger exponents
    x1 = np.array([2.0, 3.0], dtype=np.float32)
    x2 = np.array([5.0, 4.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting example
    x1 = np.array([1.0, 2.0], dtype=np.float32)
    x2 = np.array(2.0, dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array(2.0, dtype=np.float32)
    x2 = np.array(3.0, dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([0.5, 0.75], dtype=np.float32)
    x2 = np.array([-1.0, -2.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.float_power"] = tf_experimental_numpy_float_power_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.float_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.float_power'.")

check_valid('tf.experimental.numpy.float_power', generated_inputs['tf.experimental.numpy.float_power'], lib="tf", suffix=0)
