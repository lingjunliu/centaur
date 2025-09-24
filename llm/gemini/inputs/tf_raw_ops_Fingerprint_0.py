
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_fingerprint_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Fingerprint operation.
    """
    list_of_inputs = []
    
    # Using dtype=object to avoid specific fixed-size string dtypes that might not be supported.
    method_tensor = np.array('farmhash64', dtype=object)

    # Input 1: Basic 2D integer tensor
    input_dict_1 = {
        'data': np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        'method': method_tensor,
        'name': 'int32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D float tensor
    input_dict_2 = {
        'data': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'method': method_tensor,
        'name': 'float32_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D tensor of int64
    input_dict_3 = {
        'data': np.array([1, 1, 2, 3, 5, 8], dtype=np.int64),
        'method': method_tensor,
        'name': 'int64_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D tensor with small integers (int8)
    input_dict_4 = {
        'data': np.ones((2, 2, 2, 2), dtype=np.int8),
        'method': method_tensor,
        'name': 'int8_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean tensor
    input_dict_5 = {
        'data': np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        'method': method_tensor,
        'name': 'bool_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensor with negative values
    input_dict_6 = {
        'data': np.array([[-1, -100, -1000], [0, 1, 2]], dtype=np.int16),
        'method': method_tensor,
        'name': 'negative_int16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex number tensor (complex64)
    input_dict_7 = {
        'data': np.array([[1+2j, 3-4j], [5+6j, -7-8j]], dtype=np.complex64),
        'method': method_tensor,
        'name': 'complex64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Double precision float tensor (float64)
    input_dict_8 = {
        'data': np.random.rand(4, 5).astype(np.float64),
        'method': method_tensor,
        'name': 'float64_rand'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensor with an empty inner dimension
    input_dict_9 = {
        'data': np.zeros((5, 0), dtype=np.float32),
        'method': method_tensor,
        'name': 'empty_inner_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Complex number tensor (complex128)
    input_dict_10 = {
        'data': np.array([[1+2j, 3-4j], [5+6j, -7-8j]], dtype=np.complex128),
        'method': method_tensor,
        'name': 'complex128_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: High-rank tensor
    input_dict_11 = {
        'data': np.ones((1,1,1,1,1,10), dtype=np.int32),
        'method': method_tensor,
        'name': 'high_rank'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))


    return list_of_inputs

generated_inputs["tf.raw_ops.Fingerprint"] = tf_raw_ops_fingerprint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Fingerprint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fingerprint'.")

check_valid('tf.raw_ops.Fingerprint', generated_inputs['tf.raw_ops.Fingerprint'], lib="tf", suffix=0)
