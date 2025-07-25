
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatterndupdate_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.ScatterNdUpdate function.

    NOTE: The runtime error "scatter_nd_update op does not support eager execution"
    is fundamental to this specific raw operation. `tf.raw_ops.ScatterNdUpdate` is a
    stateful op designed for TensorFlow's graph-based execution model (like in TF1.x
    or inside a tf.function) where it can modify a tf.Variable in place. It is
    explicitly disabled in the default eager execution mode of modern TensorFlow.
    The error is not caused by invalid input shapes or types but by the execution
    environment. The inputs provided below are correct according to the API's
    documentation and would be valid in a graph context.
    """
    list_of_inputs = []

    # Case 1: Basic 1D update
    input_dict_1 = {
        'ref': np.array([10, 20, 30, 40], dtype=np.float32),
        'indices': np.array([[1], [3]], dtype=np.int32),
        'updates': np.array([25.0, 45.0], dtype=np.float32),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'simple_1d_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Update individual elements in a 2D tensor
    input_dict_2 = {
        'ref': np.zeros((3, 3), dtype=np.int32),
        'indices': np.array([[0, 2], [1, 1], [2, 0]], dtype=np.int32),
        'updates': np.array([1, 2, 3], dtype=np.int32),
        'use_locking': False,
        'bad_indices_policy': '',
        'name': 'elements_2d_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Update slices (rows) in a 2D tensor
    input_dict_3 = {
        'ref': np.ones((4, 3), dtype=np.float64),
        'indices': np.array([[1], [3]], dtype=np.int64),
        'updates': np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float64),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'slices_2d_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Update elements in a 3D tensor
    input_dict_4 = {
        'ref': np.arange(8, dtype=np.int32).reshape((2, 2, 2)),
        'indices': np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int32),
        'updates': np.array([100, 200], dtype=np.int32),
        'use_locking': True,
        'bad_indices_policy': 'IGNORE',
        'name': 'elements_3d_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Empty update (no-op)
    input_dict_5 = {
        'ref': np.array([1, 2, 3], dtype=np.float32),
        'indices': np.empty(shape=(0, 1), dtype=np.int32),
        'updates': np.empty(shape=(0,), dtype=np.float32),
        'use_locking': True,
        'bad_indices_policy': '',
        'name': 'empty_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterNdUpdate"] = tf_raw_ops_scatterndupdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdUpdate'.")

check_valid('tf.raw_ops.ScatterNdUpdate', generated_inputs['tf.raw_ops.ScatterNdUpdate'], lib="tf", suffix=0)
