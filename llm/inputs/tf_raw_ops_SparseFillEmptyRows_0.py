
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
    values = np.array(['a', 'b', 'c', 'd'], dtype=object)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array('default', dtype=object)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, empty values (corrected)
    indices = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int64)
    values = np.array(['', '', '', ''], dtype=object)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array('default', dtype=object)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, integer values
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, float values
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0.0, dtype=np.float32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, dense_shape with different values
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([10, 12], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, one empty row at the begining
    indices = np.array([[1, 1], [1, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, multiple empty rows
    indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    values = np.array([1, 3], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, all rows filled
    indices = np.array([[0, 1], [1, 0], [2, 0], [3, 1], [4,2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, small dense shape
    indices = np.array([[0, 0]], dtype=np.int64)
    values = np.array([1], dtype=np.int32)
    dense_shape = np.array([1, 1], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10, valid, indices starts from 1
    indices = np.array([[1, 1], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([2, 3, 4], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, valid, indices starts from 1
    indices = np.array([[1, 1], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([2, 3, 4], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(0, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Different default value
    indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    values = np.array([10, 30], dtype=np.int32)
    dense_shape = np.array([5, 6], dtype=np.int64)
    default_value = np.array(-1, dtype=np.int32)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape, "default_value": default_value, "name": "sparse_fill_12"}
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
