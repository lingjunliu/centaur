
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SelectV2_inputs():
    list_of_inputs = []

    # Input 1
    condition = np.array([[True, False], [False, True]])
    t = np.array([[1, 2], [3, 4]])
    e = np.array([[5, 6], [7, 8]])
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    condition = np.array([True, False, True])
    t = np.array([1.0, 2.0, 3.0])
    e = np.array([4.0, 5.0, 6.0])
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    condition = np.array(True)
    t = np.array(10)
    e = np.array(20)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    condition = np.array([False])
    t = np.array([100])
    e = np.array([200])
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    t = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    e = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    condition = np.array([True, False, True, False, True])
    t = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    e = np.array([6, 7, 8, 9, 10], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    condition = np.array([[True, False], [True, True], [False, False]])
    t = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    e = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int64)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    condition = np.array([False, True, False, True])
    t = np.array([-1, -2, -3, -4], dtype=np.int32)
    e = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    condition = np.array([[[True, False], [False, True]]])
    t = np.array([[[100, 200], [300, 400]]], dtype=np.int32)
    e = np.array([[[500, 600], [700, 800]]], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    condition = np.array([False] * 5)
    t = np.array([i for i in range(5)], dtype=np.int32)
    e = np.array([i + 10 for i in range(5)], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different data types
    condition = np.array([True, False, True], dtype=np.bool_)
    t = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    e = np.array([4.4, 5.5, 6.6], dtype=np.float32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Boolean type
    condition = np.array([True, False], dtype=np.bool_)
    t = np.array([True, False], dtype=np.bool_)
    e = np.array([False, True], dtype=np.bool_)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Empty array
    condition = np.array([], dtype=np.bool_)
    t = np.array([], dtype=np.float32)
    e = np.array([], dtype=np.float32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_v2_13"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SelectV2"] = tf_raw_ops_SelectV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SelectV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SelectV2'.")

check_valid('tf.raw_ops.SelectV2', generated_inputs['tf.raw_ops.SelectV2'], lib="tf", suffix=0)
