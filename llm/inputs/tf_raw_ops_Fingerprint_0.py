
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fingerprint_inputs():
    list_of_inputs = []

    # Input 1: Simple integer data with farmhash method
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float data with farmhash method
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String data with farmhash method
    data = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.string_)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data, "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D integer data with farmhash method
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D integer data with farmhash method
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Data with different dimensions
    data = np.array([1, 2, 3], dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": "fingerprint_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Data with zero values
    data = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger data
    data = np.random.randint(0, 100, size=(5, 5), dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    data = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    method = tf.constant("farmhash::fingerprint64")
    input_dict = {"data": data.astype(np.float32), "method": method, "name": None}
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
