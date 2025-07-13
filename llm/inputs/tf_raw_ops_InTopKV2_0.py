
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_InTopKV2_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: k = 2
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(2, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger batch size
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.3, 0.6, 0.1]], dtype=np.float32)
    targets = np.array([2, 0, 1], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different targets (int64) and k (int64)
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int64)
    k = np.array(2, dtype=np.int64)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  All targets in top k
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(3, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: k=0
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Ties in predictions
    predictions = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.3]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int32)
    k = np.array(2, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Larger number of classes
    predictions = np.array([[0.1, 0.2, 0.3, 0.4, 0.0], [0.9, 0.05, 0.02, 0.01, 0.02]], dtype=np.float32)
    targets = np.array([3, 0], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: k greater than number of classes
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(4, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
   # Input 10: name specified
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": "my_topk"}
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
