
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_squared_difference_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    name = "squared_difference_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values, int32
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    name = "squared_difference_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    name = "squared_difference_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different values, int64
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    name = "squared_difference_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex64)
    name = "squared_difference_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
    y = np.array([[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]], dtype=np.complex128)
    name = "squared_difference_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16
    x = tf.constant([1, 2, 3], dtype=tf.float32)
    x = tf.cast(x, tf.bfloat16).numpy()
    y = tf.constant([4, 5, 6], dtype=tf.float32)
    y = tf.cast(y, tf.bfloat16).numpy()
    name = "squared_difference_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    name = "squared_difference_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero values
    x = np.array([0, 0, 0], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    name = "squared_difference_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different shapes but compatible (broadcasting)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    name = "squared_difference_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SquaredDifference"] = tf_raw_ops_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SquaredDifference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SquaredDifference'.")

check_valid('tf.raw_ops.SquaredDifference', generated_inputs['tf.raw_ops.SquaredDifference'], lib="tf", suffix=0)
