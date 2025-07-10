
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic equal arrays
    a1 = tf.constant(np.array([1, 2, 3]))
    a2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic unequal arrays
    a1 = tf.constant(np.array([1, 2, 3]))
    a2 = tf.constant(np.array([1, 2, 4]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional equal arrays
    a1 = tf.constant(np.array([[1, 2], [3, 4]]))
    a2 = tf.constant(np.array([[1, 2], [3, 4]]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional unequal arrays
    a1 = tf.constant(np.array([[1, 2], [3, 4]]))
    a2 = tf.constant(np.array([[1, 2], [3, 5]]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Arrays with negative values, equal
    a1 = tf.constant(np.array([-1, -2, -3]))
    a2 = tf.constant(np.array([-1, -2, -3]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with negative values, unequal
    a1 = tf.constant(np.array([-1, -2, -3]))
    a2 = tf.constant(np.array([-1, -2, -4]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Arrays with zeros
    a1 = tf.constant(np.array([0, 0, 0]))
    a2 = tf.constant(np.array([0, 0, 0]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Arrays with different dtypes but equal values. Explicitly cast to tf.int32
    a1 = tf.constant(np.array([1, 2, 3]).astype(np.int32))
    a2 = tf.constant(np.array([1, 2, 3]).astype(np.int32))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D arrays, unequal
    a1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    a2 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 9]]]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Different dtypes float32 and int32 but same value - removed
    # Input 13: Boolean arrays, equal
    a1 = tf.constant(np.array([True, False, True]))
    a2 = tf.constant(np.array([True, False, True]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Boolean arrays, unequal
    a1 = tf.constant(np.array([True, False, True]))
    a2 = tf.constant(np.array([True, True, True]))
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.array_equal"] = tf_experimental_numpy_array_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.array_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array_equal'.")

check_valid('tf.experimental.numpy.array_equal', generated_inputs['tf.experimental.numpy.array_equal'], lib="tf", suffix=0)
