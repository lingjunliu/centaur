
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selectv2_inputs():
    list_of_inputs = []

    # Input 1
    condition = np.array([[True, False], [False, True]])
    t = np.array([[1, 2], [3, 4]], dtype=np.int32)
    e = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    condition = np.array([True, False, True])
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    e = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    condition = np.array([False, False, False])
    t = np.array([1, 2, 3], dtype=np.int32)
    e = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    condition = np.array([True, True, True])
    t = np.array([1, 2, 3], dtype=np.int64)
    e = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    condition = np.array([False, True, False], dtype=np.bool_)
    t = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    e = np.array([4+4j, 5+5j, 6+6j], dtype=np.complex64)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    t = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    e = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    condition = np.array(True)
    t = np.array(10, dtype=np.int32)
    e = np.array(20, dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    condition = np.array(False)
    t = np.array(3.14, dtype=np.float32)
    e = np.array(2.71, dtype=np.float32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    condition = np.array([[True, False], [True, True]])
    t = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    e = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    condition = np.array([True, False, True, False, True])
    t = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    e = np.array([6, 7, 8, 9, 10], dtype=np.int32)
    input_dict = {"condition": condition, "t": t, "e": e, "name": "select_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SelectV2"] = tf_raw_ops_selectv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SelectV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SelectV2'.")

check_valid('tf.raw_ops.SelectV2', generated_inputs['tf.raw_ops.SelectV2'], lib="tf", suffix=0)
