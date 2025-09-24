
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic example with default values
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = 3
    on_value = np.array(1.0, dtype=np.float32)
    off_value = np.array(0.0, dtype=np.float32)
    axis = -1
    dtype = tf.float32
    name = "one_hot_1"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different on_value and off_value, axis=0
    indices = np.array([0, 2, 1], dtype=np.int32)
    depth = 4
    on_value = np.array(5, dtype=np.int32)
    off_value = np.array(-2, dtype=np.int32)
    axis = 0
    dtype = tf.int32
    name = "one_hot_2"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix indices, axis=1
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    depth = 3
    on_value = np.array(True, dtype=np.bool_)
    off_value = np.array(False, dtype=np.bool_)
    axis = 1
    dtype = tf.bool
    name = "one_hot_3"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative indices
    indices = np.array([0, -1, 2], dtype=np.int32)
    depth = 4
    on_value = np.array(1.0, dtype=np.float64)
    off_value = np.array(0.0, dtype=np.float64)
    axis = -1
    dtype = tf.float64
    name = "one_hot_4"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D indices
    indices = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int32)
    depth = 3
    on_value = np.array(1, dtype=np.int32)
    off_value = np.array(0, dtype=np.int32)
    axis = 2
    dtype = tf.int32
    name = "one_hot_5"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: axis = 0 and dtype = tf.int64
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = 3
    on_value = np.array(1, dtype=np.int64)
    off_value = np.array(0, dtype=np.int64)
    axis = 0
    dtype = tf.int64
    name = "one_hot_6"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: scalar indices
    indices = np.array(2, dtype=np.int32)
    depth = 5
    on_value = np.array(2.0, dtype=np.float32)
    off_value = np.array(-1.0, dtype=np.float32)
    axis = -1
    dtype = tf.float32
    name = "one_hot_8"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: indices with different shape and int64 dtype
    indices = np.array([[1, 3, 2], [0, 4, 1]], dtype=np.int64)
    depth = 6
    on_value = np.array(1, dtype=np.int64)
    off_value = np.array(0, dtype=np.int64)
    axis = 1
    dtype = tf.int64
    name = "one_hot_9"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bool dtype
    indices = np.array([0, 1, 2], dtype=np.int32)
    depth = 3
    on_value = np.array(True, dtype=np.bool_)
    off_value = np.array(False, dtype=np.bool_)
    axis = -1
    dtype = tf.bool
    name = "one_hot_10"

    input_dict = {
        "indices": indices,
        "depth": depth,
        "on_value": on_value,
        "off_value": off_value,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.one_hot"] = tf_one_hot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.one_hot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.one_hot'.")

check_valid('tf.one_hot', generated_inputs['tf.one_hot'], lib="tf", suffix=0)
