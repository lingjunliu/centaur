
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_greater_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive and negative integers
    x1 = tf.constant(np.array([1, 2, 3, -4, -5]))
    x2 = tf.constant(np.array([0, 2, -1, -3, -6]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float values
    x1 = tf.constant(np.array([1.0, 2.5, 3.7, -4.2, -5.9]))
    x2 = tf.constant(np.array([0.5, 2.5, -1.0, -3.9, -5.9]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Equality case
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All values in x1 are greater than x2
    x1 = tf.constant(np.array([5, 6, 7]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All values in x1 are less than x2
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([5, 6, 7]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (x2 is scalar)
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(2)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes (x1 is scalar)
    x1 = tf.constant(2)
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[0, 3], [2, 5]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensors
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed types (int and float), should still work due to implicit casting
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([0.5, 2.0, 3.5], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.greater_equal"] = tf_experimental_numpy_greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.greater_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.greater_equal'.")

check_valid('tf.experimental.numpy.greater_equal', generated_inputs['tf.experimental.numpy.greater_equal'], lib="tf", suffix=0)
