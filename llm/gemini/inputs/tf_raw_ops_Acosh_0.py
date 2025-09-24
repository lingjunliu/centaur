
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_acosh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor with values inside and outside the valid range [1, inf]
    x1 = np.array([-2.0, -0.5, 1.0, 1.2, 200.0, 10000.0, np.inf], dtype=np.float32)
    input_dict1 = {'x': x1, 'name': 'test_1d_float32'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float64 tensor with valid values
    x2 = np.array([[1.0, 5.0, 10.0], [100.0, 500.0, 1000.0]], dtype=np.float64)
    input_dict2 = {'x': x2, 'name': 'test_2d_float64'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar (0-D) float16 value
    x3 = np.array(42.0, dtype=np.float16)
    input_dict3 = {'x': x3, 'name': 'test_scalar_float16'}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 3D float32 tensor
    x4 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict4 = {'x': x4, 'name': 'test_3d_float32'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D complex64 tensor
    x5 = np.array([1+1j, -1+0j, 0+0j, 2-3j, 1.0], dtype=np.complex64)
    input_dict5 = {'x': x5, 'name': 'test_1d_complex64'}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D complex128 tensor
    x6 = np.array([[1.0, 1+1j], [-2-2j, 100.0]], dtype=np.complex128)
    input_dict6 = {'x': x6, 'name': 'test_2d_complex128'}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Empty tensor with shape (0,)
    x7 = np.array([], dtype=np.float32)
    input_dict7 = {'x': x7, 'name': 'test_empty'}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with only 1.0
    x8 = np.ones((2, 2), dtype=np.float32)
    input_dict8 = {'x': x8, 'name': 'test_ones'}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Tensor with only infinity
    x9 = np.array([np.inf, np.inf], dtype=np.float64)
    input_dict9 = {'x': x9, 'name': 'test_inf'}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Large 1D tensor
    x10 = np.linspace(1, 100, 20, dtype=np.float32)
    input_dict10 = {'x': x10, 'name': 'test_linspace'}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: 1D tensor with values < 1, which should produce nans
    x11 = np.array([0.1, 0.5, 0.99, -10.0], dtype=np.float32)
    input_dict11 = {'x': x11, 'name': 'test_nan_output'}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: Complex tensor with some pure imaginary numbers
    x12 = np.array([0+1j, 0+2j, 3+0j], dtype=np.complex64)
    input_dict12 = {'x': x12, 'name': 'test_complex_imaginary'}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.raw_ops.Acosh"] = tf_raw_ops_acosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acosh'.")

check_valid('tf.raw_ops.Acosh', generated_inputs['tf.raw_ops.Acosh'], lib="tf", suffix=0)
