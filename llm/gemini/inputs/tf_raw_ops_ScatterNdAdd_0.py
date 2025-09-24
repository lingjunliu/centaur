
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_scatter_nd_add_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ScatterNdAdd.
    This op is stateful and requires a tf.Variable as input for `ref` in eager mode.
    Passing a numpy array will cause a RuntimeError. The generated inputs are
    structurally valid according to the documentation but are expected to fail
    in an eager execution environment.
    """
    list_of_inputs = []

    # Input 1: Basic example from docs, updating elements in a rank-1 tensor (int32).
    input_dict_1 = {
        'ref': np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        'indices': np.array([[4], [3], [1], [7]], dtype=np.int32),
        'updates': np.array([9, 10, 11, 12], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Updating slices in a rank-2 tensor (float32).
    input_dict_2 = {
        'ref': np.zeros((4, 4), dtype=np.float32),
        'indices': np.array([[0], [2]], dtype=np.int64),
        'updates': np.array([[1., 1., 1., 1.], [2., 2., 2., 2.]], dtype=np.float32),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'test_case_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Updating elements in a rank-2 tensor (float64).
    input_dict_3 = {
        'ref': np.ones((4, 4), dtype=np.float64),
        'indices': np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int32),
        'updates': np.array([10.0, 20.0, 30.0], dtype=np.float64),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Updating slices in a rank-3 tensor with negative values (int32).
    input_dict_4 = {
        'ref': np.full((2, 3, 4), -1, dtype=np.int32),
        'indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'updates': np.arange(8, dtype=np.int32).reshape(2, 4),
        'use_locking': False,
        'bad_indices_policy': 'IGNORE',
        'name': 'test_case_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: indices with rank 3, updating elements in a rank-2 tensor.
    input_dict_5 = {
        'ref': np.zeros((3, 3), dtype=np.int32),
        'indices': np.array([[[0, 0]], [[2, 2]]], dtype=np.int32),
        'updates': np.array([[5], [10]], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: indices with rank 3, updating slices in a rank-3 tensor.
    input_dict_6 = {
        'ref': np.zeros((2, 3, 4), dtype=np.float32),
        'indices': np.array([[[0], [1]]], dtype=np.int64),
        'updates': np.ones((1, 2, 3, 4), dtype=np.float32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty updates.
    input_dict_7 = {
        'ref': np.arange(10, dtype=np.int64),
        'indices': np.empty((0, 1), dtype=np.int32),
        'updates': np.empty((0,), dtype=np.int64),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Duplicated indices (adds multiple times to the same element).
    input_dict_8 = {
        'ref': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'indices': np.array([[1], [1], [3]], dtype=np.int64),
        'updates': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'test_case_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: complex64 data type.
    input_dict_9 = {
        'ref': np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j], dtype=np.complex64),
        'indices': np.array([[0], [4]], dtype=np.int32),
        'updates': np.array([10+10j, -5-5j], dtype=np.complex64),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Updating a single element in a rank-4 tensor (uint8).
    input_dict_10 = {
        'ref': np.zeros((2, 2, 2, 2), dtype=np.uint8),
        'indices': np.array([[1, 0, 1, 0]], dtype=np.int64),
        'updates': np.array([255], dtype=np.uint8),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'test_case_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdAdd"] = get_tf_raw_ops_scatter_nd_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdAdd'.")

check_valid('tf.raw_ops.ScatterNdAdd', generated_inputs['tf.raw_ops.ScatterNdAdd'], lib="tf", suffix=0)
