
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_reader_reset_inputs():
    """
    This function generates a list of inputs for tf.raw_ops.ReaderReset.
    The repeated "RuntimeError: reader_reset op does not support eager execution"
    is a fundamental limitation. This op is designed for TensorFlow's graph execution
    mode and requires a stateful 'Ref' tensor for 'reader_handle', which cannot be
    instantiated from a standard numpy array in an eager context. The generated
    inputs below are a best-effort attempt to provide syntactically correct data
    that conforms to the API's signature, even though they will fail in the eager
    testing environment. All string tensors are created with dtype=object to avoid
    secondary dtype validation errors.
    """
    list_of_inputs = []

    # Input 1: Simplest case
    input_dict_1 = {
        'reader_handle': np.array('handle_alpha', dtype=object),
        'name': 'reset_alpha'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Numeric handle string
    input_dict_2 = {
        'reader_handle': np.array('1234567890', dtype=object),
        'name': 'reset_numeric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scoped name
    input_dict_3 = {
        'reader_handle': np.array('handle_beta', dtype=object),
        'name': 'scope/reset_beta'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scoped handle string
    input_dict_4 = {
        'reader_handle': np.array('readers/textline/handle_gamma', dtype=object),
        'name': 'reset_gamma'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty string for handle
    input_dict_5 = {
        'reader_handle': np.array('', dtype=object),
        'name': 'reset_empty_handle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty string for name
    input_dict_6 = {
        'reader_handle': np.array('handle_delta', dtype=object),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Long handle string
    input_dict_7 = {
        'reader_handle': np.array('a_very_long_string_used_as_a_reader_handle_for_this_test_case', dtype=object),
        'name': 'reset_long'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1-D array for handle
    input_dict_8 = {
        'reader_handle': np.array(['handle_in_array'], dtype=object),
        'name': 'reset_from_array'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Handle with hyphens
    input_dict_9 = {
        'reader_handle': np.array('handle-with-hyphens', dtype=object),
        'name': 'reset_hyphenated'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Handle with underscores
    input_dict_10 = {
        'reader_handle': np.array('handle_with_underscores', dtype=object),
        'name': 'reset_underscored'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderReset"] = tf_raw_ops_reader_reset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderReset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReset'.")

check_valid('tf.raw_ops.ReaderReset', generated_inputs['tf.raw_ops.ReaderReset'], lib="tf", suffix=0)
