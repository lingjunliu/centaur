
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 1, 6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integers
    x1 = np.array([-1, -2, -3])
    x2 = np.array([-4, -1, -6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative integers
    x1 = np.array([-1, 2, -3])
    x2 = np.array([4, -1, -6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Floating point numbers
    x1 = np.array([1.5, 2.7, 3.9])
    x2 = np.array([4.1, 1.2, 6.8])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed integers and floats
    x1 = np.array([1, 2.5, 3])
    x2 = np.array([4.5, 1, 6.2])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[4, 1], [6, 2]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    x2 = np.array([[[4, 1], [6, 2]], [[8, 3], [10, 4]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting - x1 is a scalar
    x1 = np.array(2)
    x2 = np.array([1, 3, 5])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Broadcasting - x2 is a scalar
    x1 = np.array([1, 3, 5])
    x2 = np.array(2)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dtypes (but both numeric)
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([4.0, 1.0, 6.0], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger numbers
    x1 = np.array([1000, 2000, 3000])
    x2 = np.array([4000, 100, 6000])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative and Large numbers
    x1 = np.array([-1000, 2000, -3000])
    x2 = np.array([4000, -100, -6000])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.minimum"] = tf_experimental_numpy_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.minimum'.")

check_valid('tf.experimental.numpy.minimum', generated_inputs['tf.experimental.numpy.minimum'], lib="tf", suffix=0)
