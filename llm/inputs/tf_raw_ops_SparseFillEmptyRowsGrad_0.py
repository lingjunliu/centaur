
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseFillEmptyRowsGrad_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32 grad_values
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different reverse_index_map and float64 grad_values
    reverse_index_map = np.array([2, 0, 1], dtype=np.int64)
    grad_values = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: grad_values with larger size than reverse_index_map (N_full > N)
    reverse_index_map = np.array([0, 1], dtype=np.int64)
    grad_values = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: grad_values with complex64 type
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1.0+1j, 2.0+2j, 3.0+3j], dtype=np.complex64)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: grad_values with int32 type
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: grad_values with int64 type
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: reverse_index_map with some repeated indices
    reverse_index_map = np.array([0, 0, 1], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: larger range in reverse_index_map but still within grad_values bounds
    reverse_index_map = np.array([2, 1, 0], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: grad_values with complex128 type
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1.0+1j, 2.0+2j, 3.0+3j], dtype=np.complex128)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different size
    reverse_index_map = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    grad_values = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseFillEmptyRowsGrad"] = tf_raw_ops_SparseFillEmptyRowsGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseFillEmptyRowsGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseFillEmptyRowsGrad'.")

check_valid('tf.raw_ops.SparseFillEmptyRowsGrad', generated_inputs['tf.raw_ops.SparseFillEmptyRowsGrad'], lib="tf", suffix=0)
