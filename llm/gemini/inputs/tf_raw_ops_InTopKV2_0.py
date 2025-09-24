
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_InTopKV2_inputs():
    list_of_inputs = []

    # Input 1
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    k = np.array(2, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    predictions = np.array([[0.1, 0.2, 0.7, 0.0], [0.9, 0.05, 0.05, 0.0]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(3, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    predictions = np.array([[0.1, 0.2, 0.7, 0.0], [0.9, 0.05, 0.05, 0.0]], dtype=np.float32)
    targets = np.array([3, 1], dtype=np.int64)
    k = np.array(4, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    predictions = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]], dtype=np.float32)
    targets = np.array([0, 2], dtype=np.int32)
    k = np.array(2, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    predictions = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    k = np.array(3, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    predictions = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    targets = np.array([3], dtype=np.int32)
    k = np.array(2, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    predictions = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    targets = np.array([0], dtype=np.int64)
    k = np.array(1, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    predictions = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7]], dtype=np.float32)
    targets = np.array([0, 1, 2], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    predictions = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7]], dtype=np.float32)
    targets = np.array([2, 0, 1], dtype=np.int64)
    k = np.array(2, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "in_top_k_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InTopKV2"] = tf_raw_ops_InTopKV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InTopKV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InTopKV2'.")

check_valid('tf.raw_ops.InTopKV2', generated_inputs['tf.raw_ops.InTopKV2'], lib="tf", suffix=0)
