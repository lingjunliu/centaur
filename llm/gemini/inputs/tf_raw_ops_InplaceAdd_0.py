
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inplace_add_inputs():
    list_of_inputs = []

    # Input 1: Basic test
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Adding to multiple rows
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (int32)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Adding a row of zeros
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix
    x = np.random.rand(5, 5).astype(np.float32)
    i = np.array([1, 3], dtype=np.int32)
    v = np.random.rand(2, 5).astype(np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element matrix
    x = np.array([[1.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[2.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different index
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple additions
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([0, 0], dtype=np.int32)
    v = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Different dtype (int64)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    i = np.array([1], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.int64)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Adding to the last row
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InplaceAdd"] = tf_raw_ops_inplace_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InplaceAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InplaceAdd'.")

check_valid('tf.raw_ops.InplaceAdd', generated_inputs['tf.raw_ops.InplaceAdd'], lib="tf", suffix=0)
