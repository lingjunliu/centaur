
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_one_hot_inputs():
    list_of_inputs = []

    # Input 1: Scalar indices
    indices = np.array(1, dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices, default axis
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(2.0, dtype=np.float32)
    off_value = np.array(-1.0, dtype=np.float32)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector indices, axis=0
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(1, dtype=np.int32)
    off_value = np.array(0, dtype=np.int32)
    axis = 0
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix indices, default axis
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(5.0, dtype=np.float32)
    off_value = np.array(1.0, dtype=np.float32)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix indices, axis=1
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = 1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix indices, axis=0
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = np.array(3, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = 0
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Indices with negative values
    indices = np.array([0, -1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different on and off values, int type
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(5, dtype=np.int32)
    off_value = np.array(2, dtype=np.int32)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different on and off values, float64 type
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(5.0, dtype=np.float64)
    off_value = np.array(2.0, dtype=np.float64)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different on and off values, int64 type
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = np.array(4, dtype=np.int32)
    on_value = np.array(5, dtype=np.int64)
    off_value = np.array(2, dtype=np.int64)
    axis = -1
    input_dict = {"indices": indices, "depth": depth, "on_value": on_value, "off_value": off_value, "axis": axis, "name": None}
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
