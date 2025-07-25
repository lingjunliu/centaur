
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_mul_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ScatterMul.
    """
    list_of_inputs = []

    # Input 1: Basic 1D case with float32
    ref1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices1 = np.array([1, 3], dtype=np.int32)
    updates1 = np.array([0.5, 2.0], dtype=np.float32)
    input_dict_1 = {
        'ref': ref1,
        'indices': indices1,
        'updates': updates1,
        'use_locking': False,
        'name': 'case_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D ref with duplicates and locking
    ref2 = np.ones((4, 2), dtype=np.int32)
    indices2 = np.array([0, 2, 0], dtype=np.int32)
    updates2 = np.array([[2, 2], [3, 3], [5, 5]], dtype=np.int32)
    input_dict_2 = {
        'ref': ref2,
        'indices': indices2,
        'updates': updates2,
        'use_locking': True,
        'name': 'case_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar update with float64
    ref3 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    indices3 = np.array([0, 2], dtype=np.int64)
    updates3 = np.array(0.1, dtype=np.float64)
    input_dict_3 = {
        'ref': ref3,
        'indices': indices3,
        'updates': updates3,
        'use_locking': False,
        'name': 'case_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar index with int16
    ref4 = np.ones((3, 3), dtype=np.int16)
    indices4 = np.array(1, dtype=np.int32)
    updates4 = np.array([5, 5, 5], dtype=np.int16)
    input_dict_4 = {
        'ref': ref4,
        'indices': indices4,
        'updates': updates4,
        'use_locking': False,
        'name': 'case_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: High-rank indices
    ref5 = np.ones((5, 2), dtype=np.float32)
    indices5 = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates5 = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    input_dict_5 = {
        'ref': ref5,
        'indices': indices5,
        'updates': updates5,
        'use_locking': False,
        'name': 'case_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty indices and updates
    ref6 = np.array([1, 2, 3], dtype=np.int32)
    indices6 = np.array([], dtype=np.int32)
    updates6 = np.array([], dtype=np.int32)
    input_dict_6 = {
        'ref': ref6,
        'indices': indices6,
        'updates': updates6,
        'use_locking': False,
        'name': 'case_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8 type
    ref7 = np.array([10, 20, 30], dtype=np.uint8)
    indices7 = np.array([0, 2], dtype=np.int32)
    updates7 = np.array([2, 3], dtype=np.uint8)
    input_dict_7 = {
        'ref': ref7,
        'indices': indices7,
        'updates': updates7,
        'use_locking': True,
        'name': 'case_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: complex64 type
    ref8 = np.array([1+1j, 2+2j], dtype=np.complex64)
    indices8 = np.array([1], dtype=np.int32)
    updates8 = np.array([2j], dtype=np.complex64)
    input_dict_8 = {
        'ref': ref8,
        'indices': indices8,
        'updates': updates8,
        'use_locking': False,
        'name': 'case_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: int64 with negative numbers
    ref9 = np.array([1, -1, 1, -1], dtype=np.int64)
    indices9 = np.array([0, 1, 2, 3], dtype=np.int64)
    updates9 = np.array([-10, 10, -10, 10], dtype=np.int64)
    input_dict_9 = {
        'ref': ref9,
        'indices': indices9,
        'updates': updates9,
        'use_locking': False,
        'name': 'case_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 3D ref tensor
    ref10 = np.ones((3, 2, 2), dtype=np.float32)
    indices10 = np.array([0, 2], dtype=np.int32)
    updates10 = np.full((2, 2, 2), 5.0, dtype=np.float32)
    input_dict_10 = {
        'ref': ref10,
        'indices': indices10,
        'updates': updates10,
        'use_locking': False,
        'name': 'case_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterMul"] = tf_raw_ops_scatter_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMul'.")

check_valid('tf.raw_ops.ScatterMul', generated_inputs['tf.raw_ops.ScatterMul'], lib="tf", suffix=0)
