
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_parallel_stack_inputs():
    """
    Generates a list of valid inputs for the tf.parallel_stack function.
    Note: tf.parallel_stack is a graph-mode only API and will raise a
    RuntimeError in eager execution. These inputs are valid for graph-mode.
    """
    list_of_inputs = []

    # Input 1: Basic case with 1D integer tensors
    input_dict_1 = {
        'values': [np.array([1, 4], dtype=np.int32), np.array([2, 5], dtype=np.int32), np.array([3, 6], dtype=np.int32)],
        'name': 'stack_1d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float tensors
    input_dict_2 = {
        'values': [np.array([1.1, 4.4], dtype=np.float32), np.array([2.2, 5.5], dtype=np.float32)],
        'name': 'float_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D integer tensors (matrices)
    input_dict_3 = {
        'values': [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)],
        'name': 'stack_2d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D float tensors with negative values
    input_dict_4 = {
        'values': [np.array([[-1.5, 2.0], [-3.5, 4.0]], dtype=np.float32),
                   np.array([[-5.5, 6.0], [-7.5, 8.0]], dtype=np.float32)],
        'name': 'stack_2d_float_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensors
    input_dict_5 = {
        'values': [np.zeros((2, 3, 4), dtype=np.int64), np.ones((2, 3, 4), dtype=np.int64)],
        'name': '3d_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A longer list of 1D tensors
    input_dict_6 = {
        'values': [np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])],
        'name': 'long_list_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Tensors with a dimension of size 1
    input_dict_7 = {
        'values': [np.array([[1], [2], [3]]), np.array([[4], [5], [6]])],
        'name': 'dim_size_one_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Scalar tensors (rank-0)
    input_dict_8 = {
        'values': [np.array(10, dtype=np.int32), np.array(20, dtype=np.int32), np.array(30, dtype=np.int32)],
        'name': 'scalar_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensors with a different data type (uint8)
    input_dict_9 = {
        'values': [np.array([1, 2], dtype=np.uint8), np.array([3, 4], dtype=np.uint8)],
        'name': 'uint8_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors of complex numbers
    input_dict_10 = {
        'values': [np.array([1+2j, 3+4j], dtype=np.complex64), np.array([5+6j, 7+8j], dtype=np.complex64)],
        'name': 'complex_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: List containing a single tensor
    input_dict_11 = {
        'values': [np.array([[10, 20], [30, 40]], dtype=np.int16)],
        'name': 'single_tensor_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Tensors with a zero-sized dimension
    input_dict_12 = {
        'values': [np.empty((3, 0), dtype=np.float32), np.empty((3, 0), dtype=np.float32)],
        'name': 'empty_dim_stack'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.parallel_stack"] = get_tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)
