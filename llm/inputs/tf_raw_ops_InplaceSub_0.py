
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inplace_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.5, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple rows
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[0.5, 1.0], [1.0, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[-0.5, -1.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int32)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[1, 1]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger values
    x = np.array([[100, 200], [300, 400]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[50, 75]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element arrays
    x = np.array([[1.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.5]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different index
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[0.5, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Two rows of subtraction
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    i = np.array([0, 1], dtype=np.int32)
    v = np.array([[0.5, 1.0], [1.0, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Large int type
    x = np.array([[1000000, 2000000], [3000000, 4000000]], dtype=np.int32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[500000, 750000]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.5, 1.0]], dtype=np.float64)
    input_dict = {"x": x, "i": i, "v": v, "name": "inplace_sub_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InplaceSub"] = tf_raw_ops_inplace_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InplaceSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InplaceSub'.")

check_valid('tf.raw_ops.InplaceSub', generated_inputs['tf.raw_ops.InplaceSub'], lib="tf", suffix=0)
