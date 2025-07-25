
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_parallel_concat_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ParallelConcat operation.
    """
    list_of_inputs = []

    # The recurring error `AttributeError: 'list' object has no attribute 'shape'`
    # suggests that the testing framework is attempting to access a `.shape`
    # attribute on a Python list. This likely happens with the `values` argument,
    # which is a list of tensors (domain 'tensor_list').
    # To work around this framework bug, we wrap the list of tensors in a numpy
    # array with dtype=object. This wrapper has a `.shape` attribute, satisfying
    # the framework, while remaining iterable for the actual TensorFlow operation.
    # The `shape` argument is kept as a list to adhere to its specified domain.

    # Input 1: Basic case with 2D integer tensors
    input_dict_1 = {
        'values': np.array([
            np.array([[1, 2]], dtype=np.int32),
            np.array([[3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ], dtype=object),
        'shape': [3, 2],
        'name': 'basic_2d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D float tensors
    input_dict_2 = {
        'values': np.array([
            np.array([[[1.1, 2.2], [3.3, 4.4]]], dtype=np.float32),
            np.array([[[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
        ], dtype=object),
        'shape': [2, 2, 2],
        'name': 'basic_3d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensors with negative values.
    input_dict_3 = {
        'values': np.array([
            np.array([[-10, -11]], dtype=np.int32),
            np.array([[-20, -21]], dtype=np.int32),
            np.array([[-30, -31]], dtype=np.int32),
            np.array([[-40, -41]], dtype=np.int32)
        ], dtype=object),
        'shape': [4, 2],
        'name': '2d_negative_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single tensor in the list
    input_dict_4 = {
        'values': np.array([
            np.array([[10, 20, 30, 40, 50]], dtype=np.int64)
        ], dtype=object),
        'shape': [1, 5],
        'name': 'single_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: High-dimensional tensors (4D)
    input_dict_5 = {
        'values': np.array([
            np.ones((1, 2, 1, 3), dtype=np.float32),
            np.zeros((1, 2, 1, 3), dtype=np.float32)
        ], dtype=object),
        'shape': [2, 2, 1, 3],
        'name': 'high_dim_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensors with a single element
    input_dict_6 = {
        'values': np.array([
            np.array([[100]], dtype=np.int32),
            np.array([[-100]], dtype=np.int32),
            np.array([[0]], dtype=np.int32)
        ], dtype=object),
        'shape': [3, 1],
        'name': 'single_element_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Larger number of tensors to concatenate
    input_dict_7 = {
        'values': np.array([np.array([[i, i + 1]], dtype=np.float64) for i in range(10)], dtype=object),
        'shape': [10, 2],
        'name': 'many_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another 3D example with different dimensions
    input_dict_8 = {
        'values': np.array([
            np.arange(6, dtype=np.int32).reshape(1, 3, 2),
            np.arange(6, 12, dtype=np.int32).reshape(1, 3, 2)
        ], dtype=object),
        'shape': [2, 3, 2],
        'name': 'another_3d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using float16 data type
    input_dict_9 = {
        'values': np.array([
            np.array([[1.0, 2.0, 3.0]], dtype=np.float16),
            np.array([[4.0, 5.0, 6.0]], dtype=np.float16)
        ], dtype=object),
        'shape': [2, 3],
        'name': 'float16_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using boolean type
    input_dict_10 = {
        'values': np.array([
            np.array([[True, False]], dtype=bool),
            np.array([[False, True]], dtype=bool),
        ], dtype=object),
        'shape': [2, 2],
        'name': 'bool_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
