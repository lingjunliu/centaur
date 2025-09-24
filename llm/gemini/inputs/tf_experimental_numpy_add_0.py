
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition of two scalars
    x1 = tf.constant(np.array(5))
    x2 = tf.constant(np.array(3))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition of two 1D arrays
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([4, 5, 6]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of two 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[5, 6], [7, 8]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with negative values
    x1 = tf.constant(np.array([-1, -2, -3]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition of a scalar and a 1D array (broadcasting)
    x1 = tf.constant(np.array(2))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition of a 1D array and a 2D array (broadcasting)
    x1 = tf.constant(np.array([1, 2]))
    x2 = tf.constant(np.array([[3, 4], [5, 6]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with different data types (integers and floats)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition of two 3D arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with large numbers
    x1 = tf.constant(np.array([1000000000.0, 2000000000.0], dtype=np.float32))
    x2 = tf.constant(np.array([3000000000.0, 4000000000.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with zeros
    x1 = tf.constant(np.array([0, 0, 0]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs = {}
generated_inputs["tf.experimental.numpy.add"] = tf_experimental_numpy_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.add'.")

check_valid('tf.experimental.numpy.add', generated_inputs['tf.experimental.numpy.add'], lib="tf", suffix=0)
