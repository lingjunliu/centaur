
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    xi = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data types
    xi = [np.array([1.0, 2.0, 3.0]), np.array([4, 5, 6], dtype=np.int32)]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Arrays with different shapes (but compatible for meshgrid)
    xi = [np.array([1, 2, 3]), np.array([4])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More dimensions - Reduced from 3 to 2
    xi = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    xi = [np.array([-1, 0, 1]), np.array([-2, 2])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger arrays
    xi = [np.arange(5), np.arange(6)]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array
    xi = [np.array([True, False]), np.array([False, True])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 and Int64
    xi = [np.array([1.0, 2.0], dtype=np.float64), np.array([3, 4], dtype=np.int64)]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element arrays
    xi = [np.array([1]), np.array([2])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Test with shape (2,1) and (1,2)
    xi = [np.array([[1], [2]]), np.array([[3, 4]])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.meshgrid"] = tf_experimental_numpy_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
