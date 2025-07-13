
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inplace_update_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D update
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([10, 30], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D update
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[10, 20], [50, 60]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[30.0, 40.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More updates
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    i = np.array([0, 1, 2, 3], dtype=np.int32)
    v = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InplaceUpdate"] = tf_raw_ops_inplace_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InplaceUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InplaceUpdate'.")

check_valid('tf.raw_ops.InplaceUpdate', generated_inputs['tf.raw_ops.InplaceUpdate'], lib="tf", suffix=0)
