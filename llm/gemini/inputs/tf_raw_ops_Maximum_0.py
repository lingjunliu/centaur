
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 test
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values and different shape (broadcasting)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = np.array([-2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer type (int32)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 1, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer type (int64)
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([-2, -1, -4], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([2, 1, 4], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcasting with multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([2.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: int8
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([2, 1, 0], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)
