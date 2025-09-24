
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_stringlength_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringLength function.
    """
    list_of_inputs = []

    # Input 1: Basic 1D tensor with default unit ('BYTE')
    input_dict_1 = {
        'input': np.array(['Hello', 'TensorFlow', '!', ''], dtype=object),
        'unit': 'BYTE',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 1D tensor with unit='UTF8_CHAR'
    input_dict_2 = {
        'input': np.array(['Hello', 'TensorFlow', '!', ''], dtype=object),
        'unit': 'UTF8_CHAR',
        'name': 'utf8_char_length'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D tensor with multi-byte UTF-8 characters, unit='BYTE'
    input_dict_3 = {
        'input': np.array(['你好', '👋', 'TF', '😊'], dtype=object),
        'unit': 'BYTE',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D tensor with multi-byte UTF-8 characters, unit='UTF8_CHAR'
    input_dict_4 = {
        'input': np.array(['你好', '👋', 'TF', '😊'], dtype=object),
        'unit': 'UTF8_CHAR',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D tensor with mixed strings, default unit
    input_dict_5 = {
        'input': np.array([['a', 'bc'], ['def', 'ghij']], dtype=object),
        'unit': 'BYTE',
        'name': '2d_byte_length'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D tensor with mixed strings, unit='UTF8_CHAR'
    input_dict_6 = {
        'input': np.array([['€', '👍'], ['你好', 'TensorFlow']], dtype=object),
        'unit': 'UTF8_CHAR',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Scalar (0-D) tensor
    input_dict_7 = {
        'input': np.array('scalar', dtype=object),
        'unit': 'BYTE',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Scalar (0-D) tensor with multi-byte character
    input_dict_8 = {
        'input': np.array('🚀', dtype=object),
        'unit': 'UTF8_CHAR',
        'name': 'scalar_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty 1D tensor
    input_dict_9 = {
        'input': np.array([], dtype=object),
        'unit': 'BYTE',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty 2D tensor
    input_dict_10 = {
        'input': np.empty((0, 5), dtype=object),
        'unit': 'UTF8_CHAR',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: 3D tensor
    input_dict_11 = {
        'input': np.array([[['one'], ['two']], [['three'], ['four']]], dtype=object),
        'unit': 'BYTE',
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.StringLength"] = tf_raw_ops_stringlength_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringLength' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringLength'.")

check_valid('tf.raw_ops.StringLength', generated_inputs['tf.raw_ops.StringLength'], lib="tf", suffix=0)
