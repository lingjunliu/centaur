
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_nd_add_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterNdAdd function.
    Note: This raw op is designed for TensorFlow's graph mode and is expected to raise a RuntimeError in eager execution.
    The inputs provided are valid for a graph context where 'ref' would be a tf.Variable.
    """
    list_of_inputs = []

    # Case 1: Basic 1D addition (int32) from documentation
    input_dict_1 = {
        'ref': np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        'indices': np.array([[4], [3], [1], [7]], dtype=np.int32),
        'updates': np.array([9, 10, 11, 12], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'doc_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 1D float64 addition with negative values and locking
    input_dict_2 = {
        'ref': np.array([1.0, 2.5, -3.0, 4.2], dtype=np.float64),
        'indices': np.array([[0], [2]], dtype=np.int64),
        'updates': np.array([-5.5, 1.0], dtype=np.float64),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'float64_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D tensor, updating elements (K=P)
    input_dict_3 = {
        'ref': np.zeros((3, 4), dtype=np.float32),
        'indices': np.array([[0, 1], [2, 3], [1, 1]], dtype=np.int32),
        'updates': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': '2d_element_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 2D tensor, updating slices (rows, K<P)
    input_dict_4 = {
        'ref': np.ones((3, 4), dtype=np.int32),
        'indices': np.array([[0], [2]], dtype=np.int32),
        'updates': np.array([[5, 5, 5, 5], [10, 10, 10, 10]], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': '2d_slice_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Duplicate indices to test accumulation
    input_dict_5 = {
        'ref': np.array([0, 0, 0, 0], dtype=np.int32),
        'indices': np.array([[1], [3], [1]], dtype=np.int32),
        'updates': np.array([10, 20, 30], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'duplicate_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 3D tensor, updating slices (vectors, K=2)
    input_dict_6 = {
        'ref': np.ones((2, 2, 4), dtype=np.int32),
        'indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
        'updates': np.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': '3d_slice_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: 3D tensor, updating elements (scalars, K=3)
    input_dict_7 = {
        'ref': np.full((2, 2, 2), 10, dtype=np.int32),
        'indices': np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
        'updates': np.array([-5, 5], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': '3d_element_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Empty indices and updates
    input_dict_8 = {
        'ref': np.ones((2, 3), dtype=np.int32),
        'indices': np.empty((0, 2), dtype=np.int32),
        'updates': np.empty((0,), dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'empty_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Case 9: Bad indices with 'IGNORE' policy
    input_dict_9 = {
        'ref': np.array([1, 2, 3, 4], dtype=np.int32),
        'indices': np.array([[0], [5], [2]], dtype=np.int32),
        'updates': np.array([10, 20, 30], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': 'IGNORE',
        'name': 'bad_indices_ignore'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Case 10: Higher rank indices tensor
    input_dict_10 = {
        'ref': np.zeros((5, 5), dtype=np.float32),
        'indices': np.array([[[0], [2]], [[1], [3]]], dtype=np.int32),
        'updates': np.ones((2, 2, 5), dtype=np.float32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'high_rank_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdAdd"] = tf_raw_ops_scatter_nd_add_inputs()

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
