
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic example with axis=-1
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic example with axis=0
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(2.0, dtype=np.float32)
    off_value = np.array(-1.0, dtype=np.float32)
    axis = 0
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D indices with axis=-1
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1, dtype=np.int32)
    off_value = np.array(0, dtype=np.int32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D indices with axis=1
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float64)
    off_value = np.array(0.0, dtype=np.float64)
    axis = 1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative indices with axis=-1
    indices = np.array([-1, 0, 2], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different on_value and off_value types
    indices = np.array([0, 1], dtype=np.int32)
    depth = np.array(2, dtype=np.int32)
    on_value = np.array(5, dtype=np.int32)
    off_value = np.array(-2, dtype=np.int32)
    axis = 0
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: indices with uint8 type
    indices = np.array([0, 1, 2], dtype=np.uint8)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: indices with int64 type
    indices = np.array([0, 1, 2], dtype=np.int64)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: indices with int8 type and larger depth
    indices = np.array([0, 1], dtype=np.int8)
    depth = np.array(5, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = 0
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D indices
    indices = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    name = None
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OneHot"] = tf_raw_ops_one_hot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OneHot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OneHot'.")

check_valid('tf.raw_ops.OneHot', generated_inputs['tf.raw_ops.OneHot'], lib="tf", suffix=0)
