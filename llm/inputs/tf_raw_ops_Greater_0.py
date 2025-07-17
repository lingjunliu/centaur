
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_greater_inputs():
    list_of_inputs = []

    # Input 1: Basic test with int32
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5, 2, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with int32
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 comparison
    x = np.array([5.0, 4.0, 6.0], dtype=np.float32)
    y = np.array([5.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "greater_test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 comparison with negative values
    x = np.array([-1, 0, 1], dtype=np.int64)
    y = np.array([0, -1, 0], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "greater_test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 comparison with different shapes
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "greater_test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 comparison
    x = np.array([255, 128, 0], dtype=np.uint8)
    y = np.array([128, 64, 1], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "greater_test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two dimensional int32
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[0, 3, 2], [5, 4, 7]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with int32 and different dimensions.
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([2, 3], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half comparison
    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "greater_test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 comparison (replacing uint32 as it causes errors)
    x = np.array([1000, 2000, 3000], dtype=np.int16)
    y = np.array([500, 2500, 2000], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": "greater_test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Greater"] = tf_raw_ops_greater_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Greater'.")

check_valid('tf.raw_ops.Greater', generated_inputs['tf.raw_ops.Greater'], lib="tf", suffix=0)
