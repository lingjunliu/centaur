
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_guarantee_const_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer tensor
    input_dict = {
        'input': np.array([1, 2, 3, 4], dtype=np.int32),
        'name': 'guarantee_const_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float tensor with negative values
    input_dict = {
        'input': np.array([[-1.1, 2.2], [3.3, -4.4]], dtype=np.float32),
        'name': 'guarantee_const_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar tensor (0-D)
    input_dict = {
        'input': np.array(42, dtype=np.int64),
        'name': 'guarantee_const_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    input_dict = {
        'input': np.array([[True, False], [False, True]], dtype=np.bool_),
        'name': 'guarantee_const_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher-dimensional tensor (3-D)
    input_dict = {
        'input': np.zeros((2, 3, 4), dtype=np.float64),
        'name': 'guarantee_const_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex number tensor (complex64)
    input_dict = {
        'input': np.array([1+2j, 3-4j, 5j], dtype=np.complex64),
        'name': 'guarantee_const_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex number tensor (complex128)
    input_dict = {
        'input': np.array([[1.123+2.456j], [-3.789-4.111j]], dtype=np.complex128),
        'name': 'guarantee_const_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Empty tensor (shape with a zero dimension)
    input_dict = {
        'input': np.zeros((3, 0, 2), dtype=np.int32),
        'name': 'guarantee_const_empty_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D tensor
    input_dict = {
        'input': np.array([], dtype=np.float32),
        'name': 'guarantee_const_empty_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Unsigned integer tensor
    input_dict = {
        'input': np.array([0, 255, 128], dtype=np.uint8),
        'name': 'guarantee_const_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: String tensor (using object dtype for numpy representation)
    input_dict = {
        'input': np.array([b'hello', b'world', b'guarantee'], dtype=object),
        'name': 'guarantee_const_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: A single element tensor
    input_dict = {
        'input': np.array([-100.0], dtype=np.float32),
        'name': 'guarantee_const_single_element'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GuaranteeConst"] = tf_raw_ops_guarantee_const_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GuaranteeConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GuaranteeConst'.")

check_valid('tf.raw_ops.GuaranteeConst', generated_inputs['tf.raw_ops.GuaranteeConst'], lib="tf", suffix=0)
