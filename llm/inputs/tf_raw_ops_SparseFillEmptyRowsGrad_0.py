
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseFillEmptyRowsGrad_inputs():
    list_of_inputs = []

    # Input 1
    reverse_index_map = np.array([0, 1, 2], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    reverse_index_map = np.array([0, 2], dtype=np.int64)
    grad_values = np.array([1.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    reverse_index_map = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    reverse_index_map = np.array([0], dtype=np.int64)
    grad_values = np.array([1.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    reverse_index_map = np.array([0, 1, 2, 3], dtype=np.int64)
    grad_values = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    reverse_index_map = np.array([0, 2, 4], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    reverse_index_map = np.array([0, 1], dtype=np.int64)
    grad_values = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    reverse_index_map = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    grad_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    reverse_index_map = np.array([10, 20], dtype=np.int64)
    grad_values = np.array([i for i in range(30)], dtype=np.float32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    reverse_index_map = np.array([0, 1, 2, 3], dtype=np.int64)
    grad_values = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    reverse_index_map = np.array([0, 0, 0, 0], dtype=np.int64)
    grad_values = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    input_dict = {"reverse_index_map": reverse_index_map, "grad_values": grad_values, "name": "test11"}
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
