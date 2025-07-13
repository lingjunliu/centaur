
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_truncate_mod_inputs():
    list_of_inputs = []

    # Input 1: int32
    x = np.array([5, 12, 21], dtype=np.int32)
    y = np.array([2, 5, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64
    x = np.array([-5, 12, -21], dtype=np.int64)
    y = np.array([2, -5, 7], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32
    x = np.array([5.5, 12.2, 21.7], dtype=np.float32)
    y = np.array([2.0, 5.0, 7.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array([-5.5, 12.2, -21.7], dtype=np.float64)
    y = np.array([2.0, -5.0, 7.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16 (half)
    x = np.array([5.5, 12.2, 21.7], dtype=np.float16)
    y = np.array([2.0, 5.0, 7.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, multi-dimensional
    x = np.array([[5, 12], [21, 8]], dtype=np.int32)
    y = np.array([[2, 5], [7, 3]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64, multi-dimensional with negative values
    x = np.array([[-5, 12], [-21, 8]], dtype=np.int64)
    y = np.array([[2, -5], [7, -3]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, broadcasting
    x = np.array([[5.0, 12.0], [21.0, 8.0]], dtype=np.float32)
    y = np.array([2.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, broadcasting with negative values
    x = np.array([[-5.0, 12.0], [-21.0, 8.0]], dtype=np.float64)
    y = np.array([2.0, -5.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different name
    x = np.array([5, 12, 21], dtype=np.int32)
    y = np.array([2, 5, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "my_truncate_mod"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TruncateMod"] = tf_raw_ops_truncate_mod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TruncateMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TruncateMod'.")

check_valid('tf.raw_ops.TruncateMod', generated_inputs['tf.raw_ops.TruncateMod'], lib="tf", suffix=0)
