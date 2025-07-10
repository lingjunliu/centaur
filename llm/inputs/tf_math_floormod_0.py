
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floormod_inputs():
    list_of_inputs = []

    # Input 1: Simple integers
    x = tf.constant(np.array([5, 13, -7]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2, 5, 3]), dtype=tf.int32).numpy()
    name = "floormod_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different integers, including zero and negative
    x = tf.constant(np.array([-10, 0, 15, -5]), dtype=tf.int64).numpy()
    y = tf.constant(np.array([3, 7, -4, -2]), dtype=tf.int64).numpy()
    name = "floormod_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating point numbers
    x = tf.constant(np.array([3.14, -2.71, 1.618]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([1.0, 0.5, -0.3]), dtype=tf.float32).numpy()
    name = "floormod_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Another set of floats
    x = tf.constant(np.array([-5.0, 7.5, -2.25]), dtype=tf.float64).numpy()
    y = tf.constant(np.array([2.0, -1.5, 0.75]), dtype=tf.float64).numpy()
    name = "floormod_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional integers
    x = tf.constant(np.array([[10, 15], [20, 25]]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([[3, 4], [5, 6]]), dtype=tf.int32).numpy()
    name = "floormod_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional floats
    x = tf.constant(np.array([[-2.5, 3.5], [-4.5, 5.5]]), dtype=tf.float32).numpy()
    y = tf.constant(np.array([[1.0, -1.0], [0.5, -0.5]]), dtype=tf.float32).numpy()
    name = "floormod_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Different shapes (broadcasting)
    x = tf.constant(np.array([[10, 11, 12], [13, 14, 15]]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([2, 3, 4]), dtype=tf.int32).numpy()
    name = "floormod_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All negative values
    x = tf.constant(np.array([-5, -13, -7]), dtype=tf.int32).numpy()
    y = tf.constant(np.array([-2, -5, -3]), dtype=tf.int32).numpy()
    name = "floormod_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    x = tf.constant(np.array([250, 100, 50]), dtype=tf.uint8).numpy()
    y = tf.constant(np.array([10, 3, 7]), dtype=tf.uint8).numpy()
    name = "floormod_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large numbers
    x = tf.constant(np.array([1000000000, 2000000000]), dtype=tf.int64).numpy()
    y = tf.constant(np.array([300000, 700000]), dtype=tf.int64).numpy()
    name = "floormod_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.floormod"] = tf_math_floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.floormod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floormod'.")

check_valid('tf.math.floormod', generated_inputs['tf.math.floormod'], lib="tf", suffix=0)
