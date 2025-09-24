
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_temporary_variable_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.TemporaryVariable function.
    This operation is designed for graph-mode execution and will fail in an eager
    context, which is the default in modern TensorFlow. The inputs provided here are
    syntactically correct for the operation's signature. The RuntimeError is
    expected when this op is called eagerly.
    """
    list_of_inputs = []

    # Input 1: Basic 1D float32
    input_dict_1 = {
        'shape': [16],
        'dtype': np.float32,
        'var_name': 'temp_float_vec',
        'name': 'OpFloatVec'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32
    input_dict_2 = {
        'shape': [5, 5],
        'dtype': np.int32,
        'var_name': 'temp_int_matrix',
        'name': 'OpIntMatrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float64
    input_dict_3 = {
        'shape': [2, 4, 3],
        'dtype': np.float64,
        'var_name': '',
        'name': 'OpFloat64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D int16
    input_dict_4 = {
        'shape': [1, 2, 3, 4],
        'dtype': np.int16,
        'var_name': 'temp_int16_tensor',
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar (0D) bool
    input_dict_5 = {
        'shape': [],
        'dtype': np.bool_,
        'var_name': 'temp_bool_scalar',
        'name': 'OpBoolScalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D int64
    input_dict_6 = {
        'shape': [8, 8],
        'dtype': np.int64,
        'var_name': 'temp_int64_matrix',
        'name': 'OpInt64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D uint8 vector
    input_dict_7 = {
        'shape': [256],
        'dtype': np.uint8,
        'var_name': '',
        'name': 'OpUint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D float16 (half precision)
    input_dict_8 = {
        'shape': [10, 20],
        'dtype': np.float16,
        'var_name': 'temp_float16_matrix',
        'name': 'OpFloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 1D uint16 vector
    input_dict_9 = {
        'shape': [32],
        'dtype': np.uint16,
        'var_name': 'temp_uint16_vec',
        'name': 'OpUint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large-ish 2D tensor of int8
    input_dict_10 = {
        'shape': [64, 64],
        'dtype': np.int8,
        'var_name': 'temp_int8_large',
        'name': 'OpInt8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.TemporaryVariable"] = get_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TemporaryVariable'.")

check_valid('tf.raw_ops.TemporaryVariable', generated_inputs['tf.raw_ops.TemporaryVariable'], lib="tf", suffix=0)
