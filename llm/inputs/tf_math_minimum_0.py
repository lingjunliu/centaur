
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    name = "minimum_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int32
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 1, 4], dtype=np.int32)
    name = "minimum_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, int64
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([-2, -1, -4], dtype=np.int64)
    name = "minimum_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64, different values
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    y = np.array([2.5, 1.5, 4.5], dtype=np.float64)
    name = "minimum_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, int16
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[2, 1], [4, 3]], dtype=np.int16)
    name = "minimum_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, uint8
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    y = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]], dtype=np.uint8)
    name = "minimum_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, int8
    x = np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int8)
    y = np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int8)
    name = "minimum_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 scalar
    x = np.array(5.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    name = "minimum_11"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([3.0, 0.5], dtype=np.float16)
    name = "minimum_12"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint16
    x = np.array([10, 20], dtype=np.uint16)
    y = np.array([5, 25], dtype=np.uint16)
    name = "minimum_13"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.minimum"] = tf_math_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.minimum'.")

check_valid('tf.math.minimum', generated_inputs['tf.math.minimum'], lib="tf", suffix=0)
