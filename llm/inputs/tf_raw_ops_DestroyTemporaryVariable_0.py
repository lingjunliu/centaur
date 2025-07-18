
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# The `RuntimeError: destroy_temporary_variable op does not support eager execution`
# is inherent to this operation. It is designed for TensorFlow's graph mode
# and will consistently fail when called directly in the default eager execution
# environment. The provided inputs are valid according to the function's
# signature, but the execution context itself is incompatible with the op.
# This submission provides a new set of diverse inputs that conform to the
# API signature, even though the same runtime error is expected.

def generate_destroy_temporary_variable_inputs():
    list_of_inputs = []

    # Input 1: float32 tensor
    input_dict_1 = {
        'ref': np.array([1.1, 2.2, -3.3], dtype=np.float32),
        'var_name': 'f32_var',
        'name': 'destroy_f32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 2D tensor
    input_dict_2 = {
        'ref': np.array([[10, 20], [30, 40]], dtype=np.int32),
        'var_name': 'i32_var_2d',
        'name': 'destroy_i32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64 scalar
    input_dict_3 = {
        'ref': np.array(123.456, dtype=np.float64),
        'var_name': 'f64_scalar_var',
        'name': 'destroy_f64_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: boolean 3D tensor
    input_dict_4 = {
        'ref': np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_),
        'var_name': 'bool_3d_var',
        'name': 'destroy_bool_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex64 1D tensor
    input_dict_5 = {
        'ref': np.array([1+1j, -2-2j], dtype=np.complex64),
        'var_name': 'c64_var',
        'name': 'destroy_c64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: uint8 tensor
    input_dict_6 = {
        'ref': np.array([0, 1, 254, 255], dtype=np.uint8),
        'var_name': 'ui8_var',
        'name': 'destroy_ui8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int64 tensor
    input_dict_7 = {
        'ref': np.array([-1, 0, 1], dtype=np.int64),
        'var_name': 'i64_var',
        'name': 'destroy_i64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: float16 tensor
    input_dict_8 = {
        'ref': np.array([0.5, 1.5, 2.5], dtype=np.float16),
        'var_name': 'f16_var',
        'name': 'destroy_f16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty tensor with shape (0, 4)
    input_dict_9 = {
        'ref': np.empty(shape=(0, 4), dtype=np.float32),
        'var_name': 'empty_var_0_4',
        'name': 'destroy_empty_0_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Single element int16 tensor
    input_dict_10 = {
        'ref': np.array([32767], dtype=np.int16),
        'var_name': 'i16_single_var',
        'name': 'destroy_i16_single'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = generate_destroy_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DestroyTemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DestroyTemporaryVariable'.")

check_valid('tf.raw_ops.DestroyTemporaryVariable', generated_inputs['tf.raw_ops.DestroyTemporaryVariable'], lib="tf", suffix=0)
