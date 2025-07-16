
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fill_inputs():
    list_of_inputs = []

    # Input 1
    dims = np.array([2, 3], dtype=np.int32)
    value = np.array(9, dtype=np.int32)
    input_dict = {"dims": tf.constant(dims, dtype=tf.int32), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dims = np.array([5], dtype=np.int64)
    value = np.array(-1, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int64), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dims = np.array([1, 4, 2], dtype=np.int32)
    value = np.array(1, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int32), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dims = np.array([3, 2, 1, 2], dtype=np.int64)
    value = np.array(0, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int64), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dims = np.array([0], dtype=np.int32)
    value = np.array(0, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int32), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dims = np.array([10], dtype=np.int64)
    value = np.array(1, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int64), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dims = np.array([2, 2, 2, 2, 2], dtype=np.int32)
    value = np.array(7, dtype=tf.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int32), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dims = np.array([1, 1, 1, 1], dtype=np.int64)
    value = np.array(1, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int64), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    dims = np.array([1, 2], dtype=np.int32)
    value = np.array(1000, dtype=np.int32)
    input_dict = {"dims":  tf.constant(dims, dtype=tf.int32), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dims = np.array([1], dtype=np.int64)
    value = np.array(-5, dtype=np.int32)
    input_dict = {"dims": tf.constant(dims, dtype=tf.int64), "value": tf.constant(value, dtype=tf.int32), "name": "fill_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Fill"] = tf_raw_ops_fill_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Fill' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fill'.")

check_valid('tf.raw_ops.Fill', generated_inputs['tf.raw_ops.Fill'], lib="tf", suffix=0)
