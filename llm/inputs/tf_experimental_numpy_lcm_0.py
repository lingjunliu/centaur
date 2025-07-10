
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_lcm_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x1 = tf.constant(np.array([2, 4, 6]))
    x2 = tf.constant(np.array([3, 5, 7]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger integers
    x1 = tf.constant(np.array([12, 18, 24]))
    x2 = tf.constant(np.array([30, 36, 42]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes
    x1 = tf.constant(np.array([[2, 4], [6, 8]]))
    x2 = tf.constant(np.array([[3, 5], [7, 9]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalars
    x1 = tf.constant(6)
    x2 = tf.constant(8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting
    x1 = tf.constant(np.array([2, 4, 6]))
    x2 = tf.constant(4)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional arrays
    x1 = tf.constant(np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]]))
    x2 = tf.constant(np.array([[[3, 5], [7, 9]], [[11, 13], [15, 17]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large numbers
    x1 = tf.constant(np.array([2000, 4000]))
    x2 = tf.constant(np.array([3000, 5000]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Ones and Zeros
    x1 = tf.constant(np.array([1, 0, 1]))
    x2 = tf.constant(np.array([2, 3, 0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All same numbers
    x1 = tf.constant(np.array([5, 5, 5]))
    x2 = tf.constant(np.array([7, 7, 7]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Another example
    x1 = tf.constant(np.array([15, 25, 35]))
    x2 = tf.constant(np.array([45, 55, 65]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.lcm"] = tf_experimental_numpy_lcm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.lcm'.")

check_valid('tf.experimental.numpy.lcm', generated_inputs['tf.experimental.numpy.lcm'], lib="tf", suffix=0)
