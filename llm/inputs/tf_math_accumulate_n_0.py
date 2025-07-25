
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Assume generated_inputs dictionary is pre-initialized
# generated_inputs = {}

def tf_math_accumulate_n_inputs():
    """
    Generates a list of valid inputs for the tf.math.accumulate_n function.
    """
    list_of_inputs = []

    # Input 1: Basic case with 2D int32 tensors
    inputs_list_1 = [np.array([[1, 2], [3, 4]], dtype=np.int32),
                     np.array([[5, 6], [7, 8]], dtype=np.int32)]
    input_dict_1 = {
        'inputs': np.stack(inputs_list_1),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: List of three 1D float32 tensors with negative values
    inputs_list_2 = [np.array([1.5, -2.5], dtype=np.float32),
                     np.array([0.5, 0.5], dtype=np.float32),
                     np.array([-1.0, 2.0], dtype=np.float32)]
    input_dict_2 = {
        'inputs': np.stack(inputs_list_2),
        'shape': None,
        'tensor_dtype': None,
        'name': 'float_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Tensors with int64 dtype
    inputs_list_3 = [np.array([[10], [20]], dtype=np.int64),
                     np.array([[-5], [-15]], dtype=np.int64)]
    input_dict_3 = {
        'inputs': np.stack(inputs_list_3),
        'shape': None,
        'tensor_dtype': np.int64, # Specify dtype, let shape be inferred
        'name': 'sum_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Higher dimension (3D) tensors with float64
    inputs_list_4 = [np.ones((2, 2, 2), dtype=np.float64),
                     np.full((2, 2, 2), 5.5, dtype=np.float64)]
    input_dict_4 = {
        'inputs': np.stack(inputs_list_4),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_3d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single tensor in the input list
    inputs_list_5 = [np.array([100, 200, 300], dtype=np.int32)]
    input_dict_5 = {
        'inputs': np.stack(inputs_list_5),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: List of scalar (0-D) tensors
    inputs_list_6 = [np.array(10, dtype=np.int32),
                     np.array(20, dtype=np.int32),
                     np.array(-5, dtype=np.int32)]
    input_dict_6 = {
        'inputs': np.stack(inputs_list_6),
        'shape': None,
        'tensor_dtype': None,
        'name': 'scalar_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A larger list of tensors (five 2x2 matrices of ones)
    inputs_list_7 = [np.ones((2, 2), dtype=np.float32) for _ in range(5)]
    input_dict_7 = {
        'inputs': np.stack(inputs_list_7),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensors containing zeros
    inputs_list_8 = [np.array([[1, 0], [0, 4]], dtype=np.int32),
                     np.array([[0, 2], [3, 0]], dtype=np.int32),
                     np.zeros((2, 2), dtype=np.int32)]
    input_dict_8 = {
        'inputs': np.stack(inputs_list_8),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_with_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensors with a zero-sized dimension
    inputs_list_9 = [np.empty((2, 0, 3), dtype=np.float32),
                     np.empty((2, 0, 3), dtype=np.float32)]
    input_dict_9 = {
        'inputs': np.stack(inputs_list_9),
        'shape': None,
        'tensor_dtype': None,
        'name': 'sum_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex numbers (complex64)
    inputs_list_10 = [np.array([1+2j, 3-4j], dtype=np.complex64),
                      np.array([5-6j, -7+8j], dtype=np.complex64)]
    input_dict_10 = {
        'inputs': np.stack(inputs_list_10),
        'shape': None,
        'tensor_dtype': None,
        'name': 'complex_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

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
