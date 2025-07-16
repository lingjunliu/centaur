
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fingerprint_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String data
    data = np.array(["hello", "world"], dtype="S5")
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array (at least rank 1)
    data = np.array([[]], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with negative values
    data = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger array
    data = np.random.randint(0, 100, size=(10, 10), dtype=np.int32)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different data type (int64)
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another string data
    data = np.array(["test1", "test2"], dtype="S5")
    method = np.array("farmhash::fingerprint64", dtype="S23")
    input_dict = {"data": data, "method": method, "name": None}
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
