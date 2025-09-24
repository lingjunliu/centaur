
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_min_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterMin operation.
    Note: This op is designed for TensorFlow's graph mode and expects a 'ref'
    tensor, which is not directly supported in eager execution. The execution of
    these inputs in an eager context is expected to raise a RuntimeError.
    """
    list_of_inputs = []

    # Input 1: Basic 1D float32
    input_dict_1 = {
        'ref': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'indices': np.array([4, 3, 1, 0], dtype=np.int32),
        'updates': np.array([0.5, 3.5, 1.5, -1.0], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_1d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D ref, 1D indices, int32
    input_dict_2 = {
        'ref': np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[5, 25], [45, 65]], dtype=np.int32),
        'use_locking': False,
        'name': '2d_ref_1d_indices_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Duplicate indices, int64
    input_dict_3 = {
        'ref': np.array([100, 200, 300], dtype=np.int64),
        'indices': np.array([1, 0, 1, 2, 0], dtype=np.int64),
        'updates': np.array([150, 50, 180, 250, 90], dtype=np.int64),
        'use_locking': True,
        'name': 'duplicate_indices_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative values, float64
    input_dict_4 = {
        'ref': np.array([-1.5, -2.5, -3.5], dtype=np.float64),
        'indices': np.array([2, 0], dtype=np.int32),
        'updates': np.array([-3.0, -2.0], dtype=np.float64),
        'use_locking': False,
        'name': 'negative_values_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Higher rank indices
    input_dict_5 = {
        'ref': np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        'indices': np.array([[1, 3], [0, 2]], dtype=np.int32),
        'updates': np.array([[[3.1, 3.9], [7.1, 7.9]], [[1.1, 1.9], [5.1, 5.9]]], dtype=np.float32),
        'use_locking': False,
        'name': 'higher_rank_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Scalar updates
    input_dict_6 = {
        'ref': np.array([[10, 20], [30, 40]], dtype=np.int32),
        'indices': np.array([0, 1, 0], dtype=np.int32),
        'updates': np.array(15, dtype=np.int32),
        'use_locking': False,
        'name': 'scalar_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty indices and updates
    input_dict_7 = {
        'ref': np.array([1., 2., 3.], dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.float32),
        'use_locking': False,
        'name': 'empty_indices_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D ref
    input_dict_8 = {
        'ref': np.ones((3, 2, 2), dtype=np.float32) * 10,
        'indices': np.array([0, 2], dtype=np.int64),
        'updates': np.ones((2, 2, 2), dtype=np.float32) * 5,
        'use_locking': True,
        'name': '3d_ref'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single element ref
    input_dict_9 = {
        'ref': np.array([[100.0]], dtype=np.float64),
        'indices': np.array([0, 0, 0], dtype=np.int64),
        'updates': np.array([[99.0], [101.0], [98.0]], dtype=np.float64),
        'use_locking': True,
        'name': 'single_element_ref'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: All zero updates
    input_dict_10 = {
        'ref': np.array([10, -10, 20, -20], dtype=np.int32),
        'indices': np.array([0, 1, 2, 3], dtype=np.int32),
        'updates': np.array([0, 0, 0, 0], dtype=np.int32),
        'use_locking': True,
        'name': 'all_zero_updates'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterMin"] = tf_raw_ops_scatter_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMin'.")

check_valid('tf.raw_ops.ScatterMin', generated_inputs['tf.raw_ops.ScatterMin'], lib="tf", suffix=0)
