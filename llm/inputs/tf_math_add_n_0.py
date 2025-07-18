
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_add_n_inputs():
    """
    Generates a list of valid inputs for the tf.math.add_n function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D integer tensors (2 tensors)
    input_dict_1 = {
        'inputs': np.array([
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6], [7, 8]], dtype=np.int32)
        ]),
        'name': 'basic_2d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D integer tensors (3 tensors), from example
    input_dict_2 = {
        'inputs': np.array([
            np.array([[3, 5], [4, 8]], dtype=np.int32),
            np.array([[1, 6], [2, 9]], dtype=np.int32),
            np.array([[3, 5], [4, 8]], dtype=np.int32)
        ]),
        'name': 'example_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D float32 tensors with negative values
    input_dict_3 = {
        'inputs': np.array([
            np.array([-1.5, 2.0, -3.5], dtype=np.float32),
            np.array([1.0, -2.5, 3.0], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32)
        ]),
        'name': '1d_float_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D int64 tensors
    input_dict_4 = {
        'inputs': np.array([
            np.arange(8, dtype=np.int64).reshape((2, 2, 2)),
            np.arange(8, 16, dtype=np.int64).reshape((2, 2, 2))
        ]),
        'name': '3d_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 0D tensors (scalars)
    input_dict_5 = {
        'inputs': np.array([
            np.array(10, dtype=np.int32),
            np.array(20, dtype=np.int32),
            np.array(-5, dtype=np.int32)
        ]),
        'name': 'scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D float64 tensors
    input_dict_6 = {
        'inputs': np.array([
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
            np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float64)
        ]),
        'name': '2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Tensors with zeros
    input_dict_7 = {
        'inputs': np.array([
            np.array([[1, 0], [0, 1]], dtype=np.int32),
            np.array([[0, 2], [3, 0]], dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32)
        ]),
        'name': 'with_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: List with a single tensor
    input_dict_8 = {
        'inputs': np.array([
            np.array([100, 200, 300], dtype=np.int32)
        ]),
        'name': 'single_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large list of tensors (5 tensors)
    input_dict_9 = {
        'inputs': np.array([np.full((2, 3), i, dtype=np.float32) for i in range(1, 6)]),
        'name': 'large_list_of_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensors
    input_dict_10 = {
        'inputs': np.array([
            np.ones((1, 2, 2, 3), dtype=np.int32),
            np.ones((1, 2, 2, 3), dtype=np.int32) * 2,
            np.ones((1, 2, 2, 3), dtype=np.int32) * 3
        ]),
        'name': '4d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Complex numbers
    input_dict_11 = {
        'inputs': np.array([
            np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            np.array([5 - 1j, -2 + 3j], dtype=np.complex64)
        ]),
        'name': 'complex_numbers'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Name parameter is None
    input_dict_12 = {
        'inputs': np.array([
            np.array([10, 20], dtype=np.int32),
            np.array([30, 40], dtype=np.int32)
        ]),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
