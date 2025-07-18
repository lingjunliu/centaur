
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_substr_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Substr function.
    """
    list_of_inputs = []

    # Input 1: Basic case with scalar pos and len (from docs)
    input_dict_1 = {
        'input': np.array([b'Hello', b'World'], dtype=object),
        'pos': np.array(1, dtype=np.int32),
        'len': np.array(3, dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_1'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: pos and len with same shape as input (from docs)
    input_dict_2 = {
        'input': np.array([[b'ten', b'eleven', b'twelve'],
                           [b'thirteen', b'fourteen', b'fifteen'],
                           [b'sixteen', b'seventeen', b'eighteen']], dtype=object),
        'pos': np.array([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]], dtype=np.int32),
        'len': np.array([[2, 3, 4],
                         [4, 3, 2],
                         [5, 5, 5]], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_2'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Broadcasting pos and len onto input (from docs)
    input_dict_3 = {
        'input': np.array([[b'ten', b'eleven', b'twelve'],
                           [b'thirteen', b'fourteen', b'fifteen'],
                           [b'sixteen', b'seventeen', b'eighteen'],
                           [b'nineteen', b'twenty', b'twentyone']], dtype=object),
        'pos': np.array([1, 2, 3], dtype=np.int32),
        'len': np.array([1, 2, 3], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_3'
    }
    list_of_inputs.append(input_dict_3)
    
    # Input 4: Broadcasting input onto pos and len (from docs)
    input_dict_4 = {
        'input': np.array(b'thirteen', dtype=object),
        'pos': np.array([1, 5, 7], dtype=np.int32),
        'len': np.array([3, 2, 1], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_4'
    }
    list_of_inputs.append(input_dict_4)
    
    # Input 5: Negative pos (count from end) with int64 type
    input_dict_5 = {
        'input': np.array([b'TensorFlow', b'Broadcasting', b'Python'], dtype=object),
        'pos': np.array([-4, -5, -1], dtype=np.int64),
        'len': np.array([2, 3, 1], dtype=np.int64),
        'unit': 'BYTE',
        'name': 'test_negative_pos_int64'
    }
    list_of_inputs.append(input_dict_5)
    
    # Input 6: Negative len (take rest of string)
    input_dict_6 = {
        'input': np.array([b'abcdef', b'ghijkl', b'mnopqr'], dtype=object),
        'pos': np.array([2, 0, 4], dtype=np.int32),
        'len': np.array([-1, -1, -1], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_negative_len'
    }
    list_of_inputs.append(input_dict_6)
    
    # Input 7: len extends beyond string length
    input_dict_7 = {
        'input': np.array([b'short', b'longer_string'], dtype=object),
        'pos': np.array([1, 7], dtype=np.int32),
        'len': np.array([10, 20], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_long_len'
    }
    list_of_inputs.append(input_dict_7)
    
    # Input 8: unit='UTF8_CHAR'
    input_dict_8 = {
        'input': np.array([b'\xe4\xbd\xa0\xe5\xa5\xbd', # "你好"
                           b'UTF-8 \xf0\x9f\x98\x80'], dtype=object), # "UTF-8 😀"
        'pos': np.array([0, 6], dtype=np.int32),
        'len': np.array([1, 1], dtype=np.int32),
        'unit': 'UTF8_CHAR',
        'name': 'test_utf8'
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Zero length
    input_dict_9 = {
        'input': np.array([b'non-empty', b'another'], dtype=object),
        'pos': np.array([0, 3], dtype=np.int32),
        'len': np.array([0, 0], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_zero_len'
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Empty input string
    input_dict_10 = {
        'input': np.array([b'', b'not_empty', b''], dtype=object),
        'pos': np.array([0, 2, 0], dtype=np.int32),
        'len': np.array([5, 3, -1], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_empty_string'
    }
    list_of_inputs.append(input_dict_10)

    # Input 11: Broadcasting scalar input to 2D pos/len
    input_dict_11 = {
        'input': np.array(b'Broadcasting', dtype=object),
        'pos': np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32),
        'len': np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32),
        'unit': 'BYTE',
        'name': 'test_broadcast_scalar_input'
    }
    list_of_inputs.append(input_dict_11)

    # Input 12: Mixed positive and negative pos and len (Corrected)
    input_dict_12 = {
        'input': np.array([[b'alpha', b'beta'], [b'gamma', b'delta']], dtype=object),
        'pos': np.array([[1, -2], [-3, 0]], dtype=np.int32),
        'len': np.array([[3, -1], [2, 10]], dtype=np.int32), # Corrected to int32 to match pos
        'unit': 'BYTE',
        'name': 'test_mixed_values'
    }
    list_of_inputs.append(input_dict_12)

    return list_of_inputs

generated_inputs["tf.raw_ops.Substr"] = get_tf_raw_ops_substr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Substr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Substr'.")

check_valid('tf.raw_ops.Substr', generated_inputs['tf.raw_ops.Substr'], lib="tf", suffix=0)
