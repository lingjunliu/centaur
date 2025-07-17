
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fingerprint_inputs():
    list_of_inputs = []

    # Input 1: Basic integer input
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float input
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String input
    data = np.array([["hello", "world"], ["tensorflow", "rocks"]], dtype=np.string_)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D input
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D input
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (int64)
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: name
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype=np.object_)
    input_dict = {"data": data, "method": method, "name": "fingerprint_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Fingerprint"] = tf_raw_ops_fingerprint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Fingerprint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fingerprint'.")

check_valid('tf.raw_ops.Fingerprint', generated_inputs['tf.raw_ops.Fingerprint'], lib="tf", suffix=0)
