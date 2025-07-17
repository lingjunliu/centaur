
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lessequal_inputs():
    list_of_inputs = []

    # Input 1: Basic integer comparison
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float comparison
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.5, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([-2, -2, -2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 1], [4, 3]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data types (float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 1.5, 3.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data types (int64)
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([2, 1, 4], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero values
    x = np.array([0, 0, 0], dtype=np.int32)
    y = np.array([0, 1, -1], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Equal values
    x = np.array([5, 5, 5], dtype=np.int32)
    y = np.array([5, 5, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([2, 2, 2], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([2, 1, 4], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LessEqual"] = tf_raw_ops_lessequal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LessEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LessEqual'.")

check_valid('tf.raw_ops.LessEqual', generated_inputs['tf.raw_ops.LessEqual'], lib="tf", suffix=0)
