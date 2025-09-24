
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_not_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic test with integers
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([3, 2, 1]), dtype=tf.int32).numpy()
    name = "not_equal_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting test
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32).numpy()
    y = tf.constant(2, dtype=tf.int32).numpy()
    name = "not_equal_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with floats
    x = tf.constant(np.array([1.0, 2.5, 3.2]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([1.0, 2.0, 3.2]), dtype=tf.float32).numpy()
    name = "not_equal_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with different shapes (broadcasting)
    x = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([1, 4]), dtype=tf.int32).numpy()
    name = "not_equal_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with negative numbers
    x = tf.constant(np.array([-1, -2, 3]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([1, -2, -3]), dtype=tf.int32).numpy()
    name = "not_equal_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with booleans
    x = tf.constant(np.array([True, False, True]), dtype=tf.bool).numpy()
    y = tf.constant(np.array([False, False, True]), dtype=tf.bool).numpy()
    name = "not_equal_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with 3D tensors
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([[[1, 3], [3, 5]], [[5, 7], [7, 9]]]), dtype=tf.int32).numpy()
    name = "not_equal_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with 0
    x = tf.constant(np.array([0, 1, 0]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([1, 0, 0]), dtype=tf.int32).numpy()
    name = "not_equal_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Test with the same values
    x = tf.constant(np.array([5, 5, 5]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([5, 5, 5]), dtype=tf.int32).numpy()
    name = "not_equal_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Test with large numbers
    x = tf.constant(np.array([1000000, 2000000]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2000000, 1000000]), dtype=tf.int32).numpy()
    name = "not_equal_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.not_equal"] = tf_math_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.not_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.not_equal'.")

check_valid('tf.math.not_equal', generated_inputs['tf.math.not_equal'], lib="tf", suffix=0)
