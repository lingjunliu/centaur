
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_log1p_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D tensor
    x1 = np.array([0, 0.5, 1, 5, 100], dtype=np.float32)
    input_dict_1 = {'x': x1, 'name': 'float32_1d'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64 2D tensor with values near the -1 boundary
    x2 = np.array([[-0.99999, 1e-9], [0, 1e12]], dtype=np.float64)
    input_dict_2 = {'x': x2, 'name': 'float64_2d_boundary'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float16 (half) scalar
    x3 = np.array(0.125, dtype=np.float16)
    input_dict_3 = {'x': x3, 'name': 'float16_scalar'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: complex64 1D tensor
    x5 = np.array([1+1j, -0.5+0.5j, 0-1j, 10+0j], dtype=np.complex64)
    input_dict_4 = {'x': x5, 'name': 'complex64_1d'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex128 2D tensor
    x6 = np.array([[1, 2+3j], [-0.9+0.1j, 1e9+1e9j]], dtype=np.complex128)
    input_dict_5 = {'x': x6, 'name': 'complex128_2d'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty tensor
    x7 = np.array([], dtype=np.float32)
    input_dict_6 = {'x': x7, 'name': 'empty_tensor'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Tensor with a single element
    x8 = np.array([999.0], dtype=np.float64)
    input_dict_7 = {'x': x8, 'name': 'single_element_tensor'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large 2D tensor with mixed positive and negative values
    x9 = np.arange(-0.9, 9.1, 0.5, dtype=np.float32).reshape(4, 5)
    input_dict_8 = {'x': x9, 'name': 'large_mixed_2d'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex number where 1+x is purely imaginary
    x10 = np.array([-1+1j, -1-2j, -1+1e-6j], dtype=np.complex64)
    input_dict_9 = {'x': x10, 'name': 'purely_imaginary_1plusx'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 0-D (scalar) complex number
    x11 = np.array(5-5j, dtype=np.complex128)
    input_dict_10 = {'x': x11, 'name': 'complex_scalar'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Zero tensor
    x12 = np.array(0, dtype=np.float32)
    input_dict_11 = {'x': x12, 'name': 'zero_scalar'}
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: high dimensional tensor
    x13 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict_12 = {'x': x13, 'name': 'high_dim_tensor'}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.Log1p"] = get_tf_raw_ops_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log1p'.")

check_valid('tf.raw_ops.Log1p', generated_inputs['tf.raw_ops.Log1p'], lib="tf", suffix=0)
