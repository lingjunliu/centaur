
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseFillEmptyRows_inputs():
    list_of_inputs = []

    # Input 1, valid
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array(['a', 'b', 'c', 'd'], dtype=np.object_)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(b'default', dtype=np.object_)

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "default_value": default_value,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, int values
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "default_value": default_value,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, float values
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0.0, dtype=np.float32)

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "default_value": default_value,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, large dense shape
    indices = np.array([[0, 1], [9, 3], [20, 0], [31, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([100, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "default_value": default_value,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, one row
    indices = np.array([[0, 1]], dtype=np.int64)
    values = np.array([1], dtype=np.int32)
    dense_shape = np.array([1, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)

    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape,
        "default_value": default_value,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseFillEmptyRows"] = tf_raw_ops_SparseFillEmptyRows_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseFillEmptyRows' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseFillEmptyRows'.")

check_valid('tf.raw_ops.SparseFillEmptyRows', generated_inputs['tf.raw_ops.SparseFillEmptyRows'], lib="tf", suffix=0)
