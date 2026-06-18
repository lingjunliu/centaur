
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseAddGrad_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'name': "sparse_add_grad_1",
        'backprop_val_grad': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'a_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'b_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [1, 1], [1, 2]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'name': "sparse_add_grad_2",
        'backprop_val_grad': np.array([0.5, -1.5, 2.5], dtype=np.float32),
        'a_indices': np.array([[1], [3]], dtype=np.int64),
        'b_indices': np.array([[2], [3]], dtype=np.int64),
        'sum_indices': np.array([[1], [2], [3]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': "sparse_add_grad_3",
        'backprop_val_grad': np.array([10, 20], dtype=np.int32),
        'a_indices': np.array([[0, 1]], dtype=np.int64),
        'b_indices': np.array([[1, 0]], dtype=np.int64),
        'sum_indices': np.array([[0, 1], [1, 0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': "sparse_add_grad_4",
        'backprop_val_grad': np.array([], dtype=np.float64),
        'a_indices': np.empty((0, 2), dtype=np.int64),
        'b_indices': np.empty((0, 2), dtype=np.int64),
        'sum_indices': np.empty((0, 2), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': "sparse_add_grad_5",
        'backprop_val_grad': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'a_indices': np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int64),
        'sum_indices': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': "sparse_add_grad_6",
        'backprop_val_grad': np.array([1.0 + 2.0j, 2.0 - 1.0j], dtype=np.complex64),
        'a_indices': np.array([[0, 0], [0, 1]], dtype=np.int64),
        'b_indices': np.array([[0, 0], [0, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [0, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': "sparse_add_grad_7",
        'backprop_val_grad': np.array([1, 2, 3], dtype=np.int64),
        'a_indices': np.array([[10, 20], [30, 40]], dtype=np.int64),
        'b_indices': np.array([[15, 25]], dtype=np.int64),
        'sum_indices': np.array([[10, 20], [15, 25], [30, 40]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': "sparse_add_grad_8",
        'backprop_val_grad': np.array([4], dtype=np.int8),
        'a_indices': np.array([[0]], dtype=np.int64),
        'b_indices': np.array([[0]], dtype=np.int64),
        'sum_indices': np.array([[0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': "sparse_add_grad_9",
        'backprop_val_grad': np.array([1.0, 2.0], dtype=np.float32),
        'a_indices': np.array([[0, 0, 0, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 0, 0, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': "sparse_add_grad_10",
        'backprop_val_grad': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        'a_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAddGrad"] = tf_raw_ops_SparseAddGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAddGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAddGrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseAddGrad', generated_inputs['tf.raw_ops.SparseAddGrad'], lib="tf", suffix=0)
