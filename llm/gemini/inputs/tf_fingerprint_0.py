
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_fingerprint_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 array
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int64 array
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    method = 'farmhash64'
    name = "fingerprint_op"
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array
    data = np.random.rand(2, 3, 4).astype(np.float32)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D string array
    data = np.array(['hello', 'world', 'tensorflow'], dtype=np.string_)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D bool array
    data = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 array with negative values
    data = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array with zeros
    data = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int16 array
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with different dtypes
    data = np.array([[1.0, 2], [3, 4.0]], dtype=np.float32)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large random array
    data = np.random.rand(100, 100).astype(np.float32)
    method = 'farmhash64'
    name = None
    input_dict = {"data": tf.convert_to_tensor(data).numpy(), "method": method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.fingerprint"] = tf_fingerprint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.fingerprint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.fingerprint'.")

check_valid('tf.fingerprint', generated_inputs['tf.fingerprint'], lib="tf", suffix=0)
