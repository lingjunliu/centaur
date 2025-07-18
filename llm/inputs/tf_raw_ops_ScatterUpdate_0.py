
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_raw_ops_scatter_update_inputs():
    """
    Generates a list of syntactically and semantically valid inputs for the
    tf.raw_ops.ScatterUpdate operation based on its documentation.

    **CRITICAL NOTE:** The `tf.raw_ops.ScatterUpdate` operation is designed for
    TensorFlow's graph execution mode and operates on mutable variable references.
    It is **fundamentally incompatible with eager execution**, which is the
    default in modern TensorFlow. Any attempt to call this raw op directly in an
    eager context will result in a `RuntimeError`, regardless of the validity of
    the inputs. The error "scatter_update op does not support eager execution"
    is a direct consequence of this incompatibility and is not caused by the
    input values provided below.

    These inputs are correct for a graph-based environment (e.g., inside a
    `tf.function` that properly handles a `tf.Variable`).
    """
    list_of_inputs = []

    # Case 1: Simple 1D update on a float32 vector.
    # updates.shape = indices.shape + ref.shape[1:] -> (2,) = (2,) + ()
    input_dict_1 = {
        'ref': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'indices': np.array([1, 4], dtype=np.int32),
        'updates': np.array([10.0, 20.0], dtype=np.float32),
        'use_locking': True,
        'name': 'graph_mode_valid_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Update slices in a 2D int32 matrix.
    # updates.shape = indices.shape + ref.shape[1:] -> (2, 3) = (2,) + (3,)
    input_dict_2 = {
        'ref': np.zeros((4, 3), dtype=np.int32),
        'indices': np.array([0, 3], dtype=np.int64),
        'updates': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'use_locking': False,
        'name': 'graph_mode_valid_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Scalar update (updates.shape = []) broadcasted to slices of a 2D matrix.
    input_dict_3 = {
        'ref': np.ones((5, 2), dtype=np.float32),
        'indices': np.array([0, 2, 4], dtype=np.int32),
        'updates': np.array(-1.0, dtype=np.float32), # Shape is ()
        'use_locking': True,
        'name': 'graph_mode_valid_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Update on a 3D tensor with a single index.
    # updates.shape = indices.shape + ref.shape[1:] -> (1, 2, 2) = (1,) + (2, 2)
    input_dict_4 = {
        'ref': np.zeros((3, 2, 2), dtype=np.float64),
        'indices': np.array([1], dtype=np.int32),
        'updates': np.ones((1, 2, 2), dtype=np.float64) * 5.0,
        'use_locking': True,
        'name': 'graph_mode_valid_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 2D indices updating a 1D tensor.
    # updates.shape = indices.shape + ref.shape[1:] -> (2, 2) = (2, 2) + ()
    input_dict_5 = {
        'ref': np.zeros(10, dtype=np.float32),
        'indices': np.array([[1, 8], [3, 6]], dtype=np.int32),
        'updates': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        'use_locking': False,
        'name': 'graph_mode_valid_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Empty indices and updates. This is a valid edge case.
    input_dict_6 = {
        'ref': np.array([1, 2, 3], dtype=np.int32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.int32),
        'use_locking': True,
        'name': 'graph_mode_valid_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Duplicate indices. The behavior is undefined but the call is valid.
    input_dict_7 = {
        'ref': np.zeros(5, dtype=np.int32),
        'indices': np.array([1, 1, 3, 3, 1], dtype=np.int32),
        'updates': np.array([10, 20, 30, 40, 50], dtype=np.int32),
        'use_locking': True,
        'name': 'graph_mode_valid_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterUpdate"] = get_raw_ops_scatter_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterUpdate'.")

check_valid('tf.raw_ops.ScatterUpdate', generated_inputs['tf.raw_ops.ScatterUpdate'], lib="tf", suffix=0)
