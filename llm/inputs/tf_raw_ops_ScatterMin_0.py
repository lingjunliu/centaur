
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_scatter_min_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterMin function.
    The 'ref' input is provided as a numpy array as per the signature requirement.
    The execution environment is expected to convert this to a tf.Variable and
    run the operation in a graph context, as tf.raw_ops.ScatterMin is not
    supported in eager execution.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32
    input_dict_1 = {
        'use_locking': False,
        'name': 'basic_float32',
        'ref': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'indices': np.array([1, 4], dtype=np.int32),
        'updates': np.array([0.5, 4.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with int32 and use_locking=True
    input_dict_2 = {
        'use_locking': True,
        'name': 'basic_int32',
        'ref': np.array([10, 20, 30, 40], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([5, 35], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64 and duplicate indices
    input_dict_3 = {
        'use_locking': False,
        'name': 'duplicate_float64',
        'ref': np.array([100.0, 200.0, 300.0], dtype=np.float64),
        'indices': np.array([0, 2, 0], dtype=np.int64),
        'updates': np.array([90.0, 250.0, 80.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: int64 and negative numbers
    input_dict_4 = {
        'use_locking': False,
        'name': 'negative_int64',
        'ref': np.array([0, -10, 5, -15], dtype=np.int64),
        'indices': np.array([1, 3], dtype=np.int64),
        'updates': np.array([-5, -20], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D ref tensor
    input_dict_5 = {
        'use_locking': False,
        'name': '2d_ref',
        'ref': np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[5, 25], [55, 5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D ref with duplicate indices
    input_dict_6 = {
        'use_locking': True,
        'name': '2d_ref_duplicates',
        'ref': np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32),
        'indices': np.array([1, 0, 1], dtype=np.int32),
        'updates': np.array([[35, 35], [5, 5], [25, 25]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3D ref tensor
    input_dict_7 = {
        'use_locking': False,
        'name': '3d_ref',
        'ref': np.full((4, 2, 2), 100, dtype=np.int32),
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.full((2, 2, 2), 50, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D indices
    input_dict_8 = {
        'use_locking': False,
        'name': '2d_indices',
        'ref': np.arange(10, dtype=np.float32),
        'indices': np.array([[1, 8], [4, 7]], dtype=np.int32),
        'updates': np.array([[1.1, 8.8], [4.4, 7.7]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Scalar update
    input_dict_9 = {
        'use_locking': False,
        'name': 'scalar_update',
        'ref': np.array([100.0, 200.0, 300.0], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array(50.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: No change case
    input_dict_10 = {
        'use_locking': False,
        'name': 'no_change',
        'ref': np.array([1, 2, 3, 4, 5], dtype=np.int32),
        'indices': np.array([0, 2, 4], dtype=np.int32),
        'updates': np.array([10, 30, 50], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterMin"] = get_tf_raw_ops_scatter_min_inputs()

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
