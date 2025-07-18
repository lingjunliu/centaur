
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_refselect_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RefSelect operation.
    This op is not compatible with eager execution, so these inputs are expected
    to raise a RuntimeError in a TF2 eager context, but are valid for graph mode.
    The 'inputs' parameter (a list of tensors) is provided as a single stacked
    numpy array to be compatible with the test harness.
    """
    list_of_inputs = []

    # Case 1: Minimal valid input - select the only element
    input_dict_1 = {
        'index': np.array(0, dtype=np.int32),
        'inputs': np.array([[10, 20, 30]], dtype=np.int32),
        'name': 'refselect_minimal_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Select the second element from a list of two 1D float tensors
    input_dict_2 = {
        'index': np.array(1, dtype=np.int32),
        'inputs': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        'name': 'refselect_float_1d_idx1_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Select the first element from a list of two 2D int tensors
    input_dict_3 = {
        'index': np.array(0, dtype=np.int32),
        'inputs': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        'name': 'refselect_int_2d_idx0_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Select the last element from a longer list
    input_dict_4 = {
        'index': np.array(3, dtype=np.int32),
        'inputs': np.arange(20, dtype=np.int64).reshape(4, 5),
        'name': 'refselect_long_list_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Select from 3D tensors
    input_dict_5 = {
        'index': np.array(1, dtype=np.int32),
        'inputs': np.random.rand(2, 2, 2, 2).astype(np.float64),
        'name': 'refselect_float64_3d_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Using bool dtype
    input_dict_6 = {
        'index': np.array(0, dtype=np.int32),
        'inputs': np.array([[True, False], [False, True]], dtype=bool),
        'name': 'refselect_bool_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Using scalar tensors (represented as shape (N,))
    input_dict_7 = {
        'index': np.array(2, dtype=np.int32),
        'inputs': np.array([100, 200, 300, 400], dtype=np.int16),
        'name': 'refselect_scalars_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Case 8: Using uint8
    input_dict_8 = {
        'index': np.array(1, dtype=np.int32),
        'inputs': np.array([[[255, 0]], [[128, 64]]], dtype=np.uint8),
        'name': 'refselect_uint8_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Case 9: Select first from a list of 3
    input_dict_9 = {
        'index': np.array(0, dtype=np.int32),
        'inputs': np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32),
        'name': 'refselect_list_of_3_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Larger tensors
    input_dict_10 = {
        'index': np.array(1, dtype=np.int32),
        'inputs': np.ones((2, 10, 10), dtype=np.float32),
        'name': 'refselect_large_tensors_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Complex64
    input_dict_11 = {
        'index': np.array(0, dtype=np.int32),
        'inputs': np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64),
        'name': 'refselect_complex64_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Case 12: Complex128
    input_dict_12 = {
        'index': np.array(1, dtype=np.int32),
        'inputs': np.array([[[1+1j]], [[2+2j]]], dtype=np.complex128),
        'name': 'refselect_complex128_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.RefSelect"] = get_refselect_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSelect' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSelect'.")

check_valid('tf.raw_ops.RefSelect', generated_inputs['tf.raw_ops.RefSelect'], lib="tf", suffix=0)
