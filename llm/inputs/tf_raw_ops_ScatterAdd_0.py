
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_add_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ScatterAdd.
    NOTE: This op is designed for TensorFlow's graph mode and requires a
    tf.Variable for the 'ref' input. Providing NumPy arrays will likely cause a
    RuntimeError in an eager execution environment, as the op cannot handle
    standard Tensors for its 'ref' argument. The generated inputs are
    syntactically correct according to the documentation but may fail at runtime
    depending on the execution context.
    """
    list_of_inputs = []

    # Case 1: Basic 1D update, float32
    input_dict_1 = {
        'ref': np.array([1, 2, 3, 4, 5], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.array([10.0, 20.0], dtype=np.float32),
        'use_locking': False,
        'name': "case_1_basic_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 1D with duplicate indices, int32 (contributions should be added)
    input_dict_2 = {
        'ref': np.array([0, 0, 0, 0], dtype=np.int32),
        'indices': np.array([1, 2, 1, 3], dtype=np.int64),
        'updates': np.array([10, 20, 5, 30], dtype=np.int32),
        'use_locking': True,
        'name': "case_2_duplicate_indices"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Basic 2D update, float64
    input_dict_3 = {
        'ref': np.zeros((4, 3), dtype=np.float64),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        'use_locking': False,
        'name': "case_3_basic_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: High-rank indices (2D) for a 1D ref
    input_dict_4 = {
        'ref': np.array([10, 20, 30, 40, 50], dtype=np.uint8),
        'indices': np.array([[0, 2], [4, 1]], dtype=np.int32),
        'updates': np.array([[1, 2], [3, 4]], dtype=np.uint8),
        'use_locking': False,
        'name': "case_4_high_rank_indices"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Scalar update (broadcast)
    input_dict_5 = {
        'ref': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'indices': np.array([0, 1], dtype=np.int32),
        'updates': np.array(10.0, dtype=np.float32),
        'use_locking': False,
        'name': "case_5_scalar_broadcast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Empty indices and updates (should be a no-op)
    input_dict_6 = {
        'ref': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.empty((0, 2), dtype=np.float32),
        'use_locking': False,
        'name': "case_6_empty_update"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Complex numbers, complex64
    input_dict_7 = {
        'ref': np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64),
        'indices': np.array([0, 1, 0], dtype=np.int32),
        'updates': np.array([[10j, -10j], [5, -5], [1, 1]], dtype=np.complex64),
        'use_locking': False,
        'name': "case_7_complex"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: 3D ref tensor update
    input_dict_8 = {
        'ref': np.zeros((3, 2, 2), dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.ones((2, 2, 2), dtype=np.float32),
        'use_locking': False,
        'name': "case_8_3d_ref"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Updating the whole ref tensor
    input_dict_9 = {
        'ref': np.zeros(5, dtype=np.int8),
        'indices': np.arange(5, dtype=np.int32),
        'updates': np.array([10, 20, 30, 40, 50], dtype=np.int8),
        'use_locking': False,
        'name': "case_9_full_update"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Using uint16 type
    input_dict_10 = {
        'ref': np.zeros(5, dtype=np.uint16),
        'indices': np.array([4, 1, 3], dtype=np.int64),
        'updates': np.array([1000, 2000, 3000], dtype=np.uint16),
        'use_locking': False,
        'name': "case_10_uint16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterAdd"] = tf_raw_ops_scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterAdd'.")

check_valid('tf.raw_ops.ScatterAdd', generated_inputs['tf.raw_ops.ScatterAdd'], lib="tf", suffix=0)
