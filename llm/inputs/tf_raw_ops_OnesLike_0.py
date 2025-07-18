
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_oneslike_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.OnesLike operation.
    """
    list_of_inputs = []

    # Input 1: Basic float32 2D tensor
    input_dict_1 = {
        'x': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'name': 'float32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D int32 tensor with negative values
    input_dict_2 = {
        'x': np.array([-1, -2, -3, 0, 1, 2, 3], dtype=np.int32),
        'name': 'int32_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D bool tensor
    input_dict_3 = {
        'x': np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_),
        'name': 'bool_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 0D (scalar) int64 tensor
    input_dict_4 = {
        'x': np.array(100, dtype=np.int64),
        'name': 'int64_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: float64 tensor
    input_dict_5 = {
        'x': np.random.rand(3, 3).astype(np.float64),
        'name': 'float64_random'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: complex64 tensor
    input_dict_6 = {
        'x': np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        'name': 'complex64_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: complex128 tensor
    input_dict_7 = {
        'x': np.array([[1.5+2.5j, -3.5-4.5j]], dtype=np.complex128),
        'name': 'complex128_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: uint8 tensor
    input_dict_8 = {
        'x': np.array([0, 127, 255], dtype=np.uint8),
        'name': 'uint8_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-dimensional tensor (5D) of int16
    input_dict_9 = {
        'x': np.zeros((1, 2, 1, 3, 1), dtype=np.int16),
        'name': 'int16_5d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with a zero dimension, changed dtype to float32 to fix error
    input_dict_10 = {
        'x': np.empty((3, 0, 2), dtype=np.float32),
        'name': 'float32_zero_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Empty tensor (shape (0,))
    input_dict_11 = {
        'x': np.array([], dtype=np.int8),
        'name': 'int8_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Large int64 tensor, changed from uint64 to fix error
    input_dict_12 = {
        'x': np.array([[10**18]], dtype=np.int64),
        'name': 'int64_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_oneslike_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OnesLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OnesLike'.")

check_valid('tf.raw_ops.OnesLike', generated_inputs['tf.raw_ops.OnesLike'], lib="tf", suffix=0)
