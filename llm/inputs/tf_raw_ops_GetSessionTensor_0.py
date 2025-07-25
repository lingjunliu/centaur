
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_get_session_tensor_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.GetSessionTensor function.
    The `FailedPreconditionError` is inherent to this op when run in isolation,
    as it requires a session state to be populated first (e.g., by another op like
    GetSessionHandle). The provided inputs are syntactically valid according to the
    API's signature but are expected to fail at runtime in an environment where
    the session state is not initialized with the specified handles.
    """
    list_of_inputs = []

    # A diverse set of dtypes to be requested.
    dtypes_to_test = [
        np.float32, np.int32, np.bool_, np.complex64, np.string_,
        np.float64, np.uint8, np.int64, np.complex128, np.int16
    ]

    for i, dtype in enumerate(dtypes_to_test):
        # Create a unique handle and name for each case.
        handle_str = f'test_handle_{i}'
        name_str = f'get_tensor_op_{i}'

        input_dict = {
            # Handle is a 0D tensor of type string. `dtype=object` is used to
            # create a NumPy array containing a Python string, which is
            # correctly interpreted as a tf.string tensor by TensorFlow.
            'handle': np.array(handle_str, dtype=object),
            'dtype': dtype,
            'name': name_str
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 11: Add a case with an empty name string
    input_dict_empty_name = {
        'handle': np.array('handle_with_empty_name', dtype=object),
        'dtype': np.uint32,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_empty_name))

    # Case 12: Add a case with a name containing slashes (like a scope)
    input_dict_scoped_name = {
        'handle': np.array('handle_with_scoped_name', dtype=object),
        'dtype': np.uint64,
        'name': 'my_scope/my_op_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_scoped_name))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionTensor"] = get_tf_raw_ops_get_session_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionTensor'.")

check_valid('tf.raw_ops.GetSessionTensor', generated_inputs['tf.raw_ops.GetSessionTensor'], lib="tf", suffix=0)
