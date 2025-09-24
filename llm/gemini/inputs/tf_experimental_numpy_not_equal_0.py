
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_not_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic comparison of two 1D tensors
    x1 = tf.constant(np.array([1, 2, 3, 4, 5]))
    x2 = tf.constant(np.array([1, 4, 3, 2, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Comparison with different data types (int and float)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array([1.0, 2.5, 3.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Comparison of two 2D tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[1, 4], [5, 4]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Comparison with a 3D tensor
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[1, 2], [3, 5]], [[5, 7], [7, 8]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Comparison with negative values
    x1 = tf.constant(np.array([-1, -2, 3]))
    x2 = tf.constant(np.array([1, -2, -3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Comparison with zero values
    x1 = tf.constant(np.array([0, 1, 0]))
    x2 = tf.constant(np.array([1, 0, 0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting example (x2 is a scalar)
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(2)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Broadcasting example (x1 is a scalar)
    x1 = tf.constant(2)
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes that are broadcastable
    x1 = tf.constant(np.array([[1, 2, 3]]))
    x2 = tf.constant(np.array([1, 2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean tensors
    x1 = tf.constant(np.array([True, False, True]))
    x2 = tf.constant(np.array([False, True, True]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty Tensors
    x1 = tf.constant(np.array([]))
    x2 = tf.constant(np.array([]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.not_equal"] = tf_experimental_numpy_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.not_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.not_equal'.")

check_valid('tf.experimental.numpy.not_equal', generated_inputs['tf.experimental.numpy.not_equal'], lib="tf", suffix=0)
