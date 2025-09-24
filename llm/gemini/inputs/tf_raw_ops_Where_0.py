
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensor
    condition = np.array([[True, False], [False, True]])
    input_dict = {"condition": condition.astype(np.bool_), "name": "where_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D boolean tensor
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_dict = {"condition": condition.astype(np.bool_), "name": "where_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D boolean tensor
    condition = np.array([True, False, True, False, True])
    input_dict = {"condition": condition.astype(np.bool_), "name": "where_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float tensor with non-zero values
    condition = np.array([[1.0, 0.0], [0.0, 2.5]])
    input_dict = {"condition": condition.astype(np.float32), "name": "where_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer tensor with zero and non-zero values
    condition = np.array([[1, 0], [0, 5]])
    input_dict = {"condition": condition.astype(np.int32), "name": "where_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensor
    condition = np.array([[1 + 1j, 0], [0, 2 - 1j]])
    input_dict = {"condition": condition.astype(np.complex64), "name": "where_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty array
    condition = np.array([])
    input_dict = {"condition": condition.astype(np.bool_), "name": "where_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Multi-dimensional float
    condition = np.random.rand(2,3,4).astype(np.float64)
    input_dict = {"condition": condition, "name": "where_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    condition = np.array([[1, 0], [2, 0]], dtype=np.uint8)
    input_dict = {"condition": condition, "name": "where_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: int64
    condition = np.array([[1, 0], [-2, 0]], dtype=np.int64)
    input_dict = {"condition": condition, "name": "where_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Where'.")

check_valid('tf.raw_ops.Where', generated_inputs['tf.raw_ops.Where'], lib="tf", suffix=0)
