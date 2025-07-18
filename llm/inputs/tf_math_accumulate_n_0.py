
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_accumulate_n_inputs():
    """
    Generates a list of valid inputs for the tf.math.accumulate_n function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D integer arrays
    inputs_1 = [
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([[5, 6], [7, 8]], dtype=np.int32),
        np.array([[9, 10], [11, 12]], dtype=np.int32)
    ]
    input_dict_1 = {
        'inputs': np.array(inputs_1),
        'shape': [2, 2],
        'tensor_dtype': np.int32,
        'name': 'basic_2d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float arrays with negative values
    inputs_2 = [
        np.array([-1.5, 2.5], dtype=np.float32),
        np.array([3.0, -4.0], dtype=np.float32)
    ]
    input_dict_2 = {
        'inputs': np.array(inputs_2),
        'shape': [2],
        'tensor_dtype': np.float32,
        'name': 'float_1d_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D integer arrays
    inputs_3 = [
        np.ones((2, 2, 2), dtype=np.int64),
        np.full((2, 2, 2), 5, dtype=np.int64)
    ]
    input_dict_3 = {
        'inputs': np.array(inputs_3),
        'shape': [2, 2, 2],
        'tensor_dtype': np.int64,
        'name': 'int64_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar inputs (0-D tensors)
    inputs_4 = [
        np.array(10, dtype=np.int32),
        np.array(20, dtype=np.int32),
        np.array(-5, dtype=np.int32)
    ]
    input_dict_4 = {
        'inputs': np.array(inputs_4),
        'shape': [],
        'tensor_dtype': np.int32,
        'name': 'scalar_inputs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A larger list of tensors (5 tensors)
    inputs_5 = [
        np.array([1, 1], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        np.array([3, 3], dtype=np.int32),
        np.array([4, 4], dtype=np.int32),
        np.array([5, 5], dtype=np.int32)
    ]
    input_dict_5 = {
        'inputs': np.array(inputs_5),
        'shape': [2],
        'tensor_dtype': np.int32,
        'name': 'large_list_of_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 data type
    inputs_6 = [
        np.array([1.0e10, -2.0e10], dtype=np.float64),
        np.array([3.0e10, 4.0e10], dtype=np.float64)
    ]
    input_dict_6 = {
        'inputs': np.array(inputs_6),
        'shape': [2],
        'tensor_dtype': np.float64,
        'name': 'float64_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single tensor in the list
    inputs_7 = [np.array([[10, 20], [30, 40]], dtype=np.int16)]
    input_dict_7 = {
        'inputs': np.array(inputs_7),
        'shape': [2, 2],
        'tensor_dtype': np.int16,
        'name': 'single_tensor_in_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensors containing zeros
    inputs_8 = [
        np.zeros((3, 3), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
        np.full((3, 3), -1.0, dtype=np.float32)
    ]
    input_dict_8 = {
        'inputs': np.array(inputs_8),
        'shape': [3, 3],
        'tensor_dtype': np.float32,
        'name': 'tensors_with_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Higher-rank tensor (4D)
    inputs_9 = [
        np.ones((1, 2, 1, 2), dtype=np.int32),
        np.ones((1, 2, 1, 2), dtype=np.int32)
    ]
    input_dict_9 = {
        'inputs': np.array(inputs_9),
        'shape': [1, 2, 1, 2],
        'tensor_dtype': np.int32,
        'name': '4d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: uint8 data type
    inputs_10 = [
        np.array([10, 20], dtype=np.uint8),
        np.array([30, 40], dtype=np.uint8)
    ]
    input_dict_10 = {
        'inputs': np.array(inputs_10),
        'shape': [2],
        'tensor_dtype': np.uint8,
        'name': 'uint8_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.math.accumulate_n"] = get_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)
