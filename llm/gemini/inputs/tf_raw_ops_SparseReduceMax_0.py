
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_reduce_max_inputs():
    list_of_inputs = []

    # Case 1
    input_dict_1 = {
        'keep_dims': False,
        'name': "op1",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([3, 4], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2
    input_dict_2 = {
        'keep_dims': True,
        'name': "op2",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([-1, -2], dtype=np.int32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3
    input_dict_3 = {
        'keep_dims': False,
        'name': "op3",
        'input_indices': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        'input_values': np.array([5.0, 10.0], dtype=np.float64),
        'input_shape': np.array([2, 2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4
    input_dict_4 = {
        'keep_dims': True,
        'name': "op4",
        'input_indices': np.array([[0], [1], [2]], dtype=np.int64),
        'input_values': np.array([10, 20, 30], dtype=np.int64),
        'input_shape': np.array([5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5
    input_dict_5 = {
        'keep_dims': False,
        'name': "op5",
        'input_indices': np.array([[0, 1]], dtype=np.int64),
        'input_values': np.array([42], dtype=np.int32),
        'input_shape': np.array([3, 3], dtype=np.int64),
        'reduction_axes': np.array([0, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6
    input_dict_6 = {
        'keep_dims': False,
        'name': "op6",
        'input_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'input_values': np.array([1, 2, 3], dtype=np.uint8),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7
    input_dict_7 = {
        'keep_dims': True,
        'name': "op7",
        'input_indices': np.array([[0, 0, 0]], dtype=np.int64),
        'input_values': np.array([0.5], dtype=np.float32),
        'input_shape': np.array([1, 1, 1], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8
    input_dict_8 = {
        'keep_dims': False,
        'name': "op8",
        'input_indices': np.array([[0, 1, 2, 3]], dtype=np.int64),
        'input_values': np.array([100], dtype=np.int16),
        'input_shape': np.array([5, 5, 5, 5], dtype=np.int64),
        'reduction_axes': np.array([1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9
    input_dict_9 = {
        'keep_dims': False,
        'name': "op9",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1, 2], dtype=np.int8),
        'input_shape': np.array([5, 5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10
    input_dict_10 = {
        'keep_dims': True,
        'name': "op10",
        'input_indices': np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
        'input_values': np.array([1.23, 4.56], dtype=np.float64),
        'input_shape': np.array([3, 3, 3], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceMax"] = tf_raw_ops_sparse_reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseReduceMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceMax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseReduceMax', generated_inputs['tf.raw_ops.SparseReduceMax'], lib="tf", suffix=0)
