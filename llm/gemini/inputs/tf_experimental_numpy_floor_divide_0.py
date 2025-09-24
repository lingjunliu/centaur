
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_floor_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x1 = tf.constant(np.array([10, 20, 30]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 7, 2]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float division
    x1 = tf.constant(np.array([10.0, 20.0, 30.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([3.0, 7.0, 2.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Division by one
    x1 = tf.constant(np.array([10, 20, 30]), dtype=tf.int32)
    x2 = tf.constant(np.array([1, 1, 1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative numbers
    x1 = tf.constant(np.array([-10, 20, -30]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, -7, 2]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes (broadcasting)
    x1 = tf.constant(np.array([[10, 20], [30, 40]]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 5]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional arrays
    x1 = tf.constant(np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[[2, 5], [3, 7]], [[5, 2], [7, 3]]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large numbers
    x1 = tf.constant(np.array([2**10, 2**9]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 7]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative floats
    x1 = tf.constant(np.array([-10.5, 20.2, -30.7]), dtype=tf.float32)
    x2 = tf.constant(np.array([3.1, -7.2, 2.3]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Division resulting in 0.0 when floored
    x1 = tf.constant(np.array([0.5, 0.9]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.0, 2.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with different shapes
    x1 = tf.constant(np.array([10, 20, 30]), dtype=tf.int32)
    x2 = tf.constant(np.array([2]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.floor_divide"] = tf_experimental_numpy_floor_divide_inputs()

for input_dict in generated_inputs["tf.experimental.numpy.floor_divide"]:
    for key in input_dict:
        input_dict[key] = input_dict[key].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.floor_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.floor_divide'.")

check_valid('tf.experimental.numpy.floor_divide', generated_inputs['tf.experimental.numpy.floor_divide'], lib="tf", suffix=0)
