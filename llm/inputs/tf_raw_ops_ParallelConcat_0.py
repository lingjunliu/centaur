
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_parallel_concat_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ParallelConcat function.
    """
    list_of_inputs = []

    # The error "AttributeError: 'list' object has no attribute 'shape'" indicates
    # that the 'shape' argument, provided as a Python list, is being treated as a
    # tensor and expected to have a .shape attribute. To fix this, we convert the
    # 'shape' list to a NumPy array, which is a tensor-like object.

    # Input 1: Basic 2D integer tensors
    input_dict = {
        'name': 'basic_2d_int32',
        'values': [np.array([[1, 2]], dtype=np.int32), np.array([[3, 4]], dtype=np.int32)],
        'shape': [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float32 tensors with negative values
    input_dict = {
        'name': '3d_float32_negative',
        'values': [
            np.array([[[-1.5, 2.0], [3.5, -4.0]]], dtype=np.float32),
            np.array([[[5.5, 6.0], [-7.5, 8.0]]], dtype=np.float32),
            np.array([[[9.5, -10.0], [11.5, 12.0]]], dtype=np.float32)
        ],
        'shape': [3, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single tensor in the list
    input_dict = {
        'name': 'single_tensor',
        'values': [np.array([[[100, 200]]], dtype=np.int64)],
        'shape': [1, 1, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of many tensors (10)
    input_dict = {
        'name': 'many_tensors_int8',
        'values': [np.array([[i]], dtype=np.int8) for i in range(-5, 5)],
        'shape': [10, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensors with more columns (float64)
    input_dict = {
        'name': 'wider_2d_tensors_float64',
        'values': [
            np.array([[1.1, 2.2, 3.3, 4.4]], dtype=np.float64),
            np.array([[5.5, 6.6, 7.7, 8.8]], dtype=np.float64)
        ],
        'shape': [2, 4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty inner dimensions
    input_dict = {
        'name': 'empty_inner_dim',
        'values': [np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)],
        'shape': [2, 0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensors
    input_dict = {
        'name': '4d_tensors_uint8',
        'values': [
            np.arange(8, dtype=np.uint8).reshape(1, 2, 2, 2),
            np.arange(8, 16, dtype=np.uint8).reshape(1, 2, 2, 2)
        ],
        'shape': [2, 2, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger number of tensors (e.g., 5) with 3D shape
    values_list = [np.random.rand(1, 3, 2).astype(np.float32) for _ in range(5)]
    input_dict = {
        'name': 'five_3d_tensors',
        'values': values_list,
        'shape': [5, 3, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Column vectors (shape [1, 1])
    input_dict = {
        'name': 'column_vectors_int16',
        'values': [np.array([[-10]], dtype=np.int16), np.array([[20]], dtype=np.int16), np.array([[-30]], dtype=np.int16)],
        'shape': [3, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensors containing zeros
    input_dict = {
        'name': 'tensors_with_zeros',
        'values': [
            np.array([[0, 1, 0]], dtype=np.int32),
            np.array([[0, 0, 0]], dtype=np.int32),
            np.array([[1, 0, 1]], dtype=np.int32)
        ],
        'shape': [3, 3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D tensor with one of the middle dimensions as 1
    input_dict = {
        'name': '4d_middle_dim_one',
        'values': [
            np.array([[[[1, 2], [3, 4]]]], dtype=np.int32),
            np.array([[[[5, 6], [7, 8]]]], dtype=np.int32)
        ],
        'shape': [2, 1, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ParallelConcat"] = get_parallel_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ParallelConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelConcat'.")

check_valid('tf.raw_ops.ParallelConcat', generated_inputs['tf.raw_ops.ParallelConcat'], lib="tf", suffix=0)
