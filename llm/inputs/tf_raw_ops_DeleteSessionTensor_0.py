
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_deletesessiontensor_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.DeleteSessionTensor.

    NOTE: This is a stateful TensorFlow operation. It requires a tensor handle that has been
    previously created and stored in the session state (e.g., via GetSessionHandle).
    In a stateless testing environment where each API call is isolated, no such handle
    exists. Therefore, calling this function will always result in a
    `FailedPreconditionError: ... DeleteSessionTensor called on null session state`.
    The inputs provided below are valid in terms of their data types and format, but
    are expected to trigger this specific runtime error, which is inherent to testing
    this stateful op in isolation.
    """
    list_of_inputs = []

    # Input 1: Basic valid handle and name
    input_dict_1 = {
        'handle': np.array(b'session_tensor_handle_1', dtype=np.string_),
        'name': 'DeleteOperation1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Name is None
    input_dict_2 = {
        'handle': np.array(b'handle_with_no_op_name', dtype=np.string_),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scoped name
    input_dict_3 = {
        'handle': np.array(b'handle_in_scope', dtype=np.string_),
        'name': 'my_scope/delete_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Name with underscores
    input_dict_4 = {
        'handle': np.array(b'handle_4', dtype=np.string_),
        'name': 'delete_op_with_underscores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Alphanumeric handle and name
    input_dict_5 = {
        'handle': np.array(b'Handle123', dtype=np.string_),
        'name': 'DeleteOp567'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty string handle
    input_dict_6 = {
        'handle': np.array(b'', dtype=np.string_),
        'name': 'DeleteWithEmptyHandleStr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Handle that looks like a path
    input_dict_7 = {
        'handle': np.array(b'/path/to/some/handle', dtype=np.string_),
        'name': 'DeleteFromPath'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Long handle and name
    input_dict_8 = {
        'handle': np.array(b'a_very_long_and_complex_handle_string_that_is_still_valid', dtype=np.string_),
        'name': 'a/very/long/and/nested/op/name/DeleteOp'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Handle with special characters
    input_dict_9 = {
        'handle': np.array(b'handle-with-hyphens-and-symbols:!@#$', dtype=np.string_),
        'name': 'DeleteOpWithSpecialHandle'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Numeric handle
    input_dict_10 = {
        'handle': np.array(b'9876543210', dtype=np.string_),
        'name': 'DeleteNumericHandleOp'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeleteSessionTensor"] = tf_raw_ops_deletesessiontensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
