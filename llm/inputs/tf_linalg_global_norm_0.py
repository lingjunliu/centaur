
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_linalg_global_norm_inputs():
    """
    Generates a list of valid inputs for the tf.linalg.global_norm function.
    """
    list_of_inputs = []

    # Input 1: Basic case with two 1-D float32 tensors
    input_dict_1 = {
        't_list': [np.array([1.0, 2.0, 3.0], dtype=np.float32),
                   np.array([4.0, 5.0], dtype=np.float32)],
        'name': 'basic_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Mixed shapes (scalar, vector, matrix) and negative values
    input_dict_2 = {
        't_list': [np.array(-5.0, dtype=np.float32),
                   np.array([1.0, -2.0], dtype=np.float32),
                   np.array([[3.0, 4.0], [-1.0, 0.0]], dtype=np.float32)],
        'name': 'mixed_shapes_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: List containing integer tensors
    input_dict_3 = {
        't_list': [np.array([1, 2, 3], dtype=np.int32),
                   np.array([[4], [5]], dtype=np.int32)],
        'name': 'integer_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: List includes a None element, which should be ignored
    input_dict_4 = {
        't_list': [np.array([3.0, 4.0], dtype=np.float32),
                   None,
                   np.array([5.0, 12.0], dtype=np.float32)],
        'name': 'with_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty list
    input_dict_5 = {
        't_list': [],
        'name': 'empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: List containing only None elements
    input_dict_6 = {
        't_list': [None, None, None],
        'name': 'only_none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Higher-dimensional tensors (3D) and float64
    input_dict_7 = {
        't_list': [np.arange(8, dtype=np.float64).reshape((2, 2, 2)),
                   np.array([-1.0, -2.0, -3.0], dtype=np.float64)],
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: List with a single tensor
    input_dict_8 = {
        't_list': [np.array([[-3.0, 4.0]], dtype=np.float32)],
        'name': 'single_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All zero tensors
    input_dict_9 = {
        't_list': [np.zeros((3, 3), dtype=np.float32),
                   np.zeros((5,), dtype=np.float32)],
        'name': 'all_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Mixed numeric types (int32, float32, float64)
    input_dict_10 = {
        't_list': [np.array([1, 2], dtype=np.int32),
                   np.array([3.0, 4.0], dtype=np.float32),
                   np.array([5.0, 6.0], dtype=np.float64)],
        'name': 'mixed_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.global_norm"] = tf_linalg_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.global_norm'.")

check_valid('tf.linalg.global_norm', generated_inputs['tf.linalg.global_norm'], lib="tf", suffix=0)
