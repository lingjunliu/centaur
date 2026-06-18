
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSlice_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'name': 'slice1',
        'indices': np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        'values': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'shape': np.array([2, 7], dtype=np.int64),
        'start': np.array([0, 0], dtype=np.int64),
        'size': np.array([2, 4], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'name': 'slice2',
        'indices': np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        'values': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'shape': np.array([2, 7], dtype=np.int64),
        'start': np.array([0, 4], dtype=np.int64),
        'size': np.array([2, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': 'slice3',
        'indices': np.array([[1], [3], [5], [8]], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.int32),
        'shape': np.array([10], dtype=np.int64),
        'start': np.array([2], dtype=np.int64),
        'size': np.array([5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': 'slice4',
        'indices': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64),
        'values': np.array([1.5, 2.5, 3.5], dtype=np.float64),
        'shape': np.array([3, 3, 3], dtype=np.int64),
        'start': np.array([1, 1, 1], dtype=np.int64),
        'size': np.array([2, 2, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': 'slice5',
        'indices': np.empty((0, 2), dtype=np.int64),
        'values': np.empty((0,), dtype=np.float32),
        'shape': np.array([5, 5], dtype=np.int64),
        'start': np.array([1, 1], dtype=np.int64),
        'size': np.array([3, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': 'slice6',
        'indices': np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64),
        'values': np.array([100, 200, 300], dtype=np.int64),
        'shape': np.array([2, 2, 2, 2], dtype=np.int64),
        'start': np.array([0, 0, 0, 0], dtype=np.int64),
        'size': np.array([1, 2, 1, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': 'slice7',
        'indices': np.array([[10], [50], [55], [60], [90]], dtype=np.int64),
        'values': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'shape': np.array([100], dtype=np.int64),
        'start': np.array([50], dtype=np.int64),
        'size': np.array([10], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': 'slice8',
        'indices': np.array([[2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.int64),
        'values': np.array([True, False, True, False], dtype=np.bool_),
        'shape': np.array([10, 10], dtype=np.int64),
        'start': np.array([2, 2], dtype=np.int64),
        'size': np.array([5, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': 'slice9',
        'indices': np.array([[1, 2, 3]], dtype=np.int64),
        'values': np.array([42], dtype=np.int32),
        'shape': np.array([5, 5, 5], dtype=np.int64),
        'start': np.array([0, 0, 0], dtype=np.int64),
        'size': np.array([5, 5, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': 'slice10',
        'indices': np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.int64),
        'shape': np.array([4, 4], dtype=np.int64),
        'start': np.array([1, 0], dtype=np.int64),
        'size': np.array([2, 4], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSlice"] = tf_raw_ops_SparseSlice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSlice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSlice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSlice', generated_inputs['tf.raw_ops.SparseSlice'], lib="tf", suffix=0)
