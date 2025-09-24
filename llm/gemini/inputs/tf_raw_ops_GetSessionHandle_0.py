
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_getsessionhandle_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.GetSessionHandle function.

    NOTE: The `tf.raw_ops.GetSessionHandle` operation is designed to work within a
    TensorFlow 1.x-style graph and session context. When executed in a standard
    TensorFlow 2.x eager context, it is expected to raise a
    `FailedPreconditionError` with the message "GetSessionHandle called on null
    session state". This error is due to the execution environment lacking the
    required session state and does not indicate that the provided inputs
    (tensor values and names) are invalid in format, type, or shape. The inputs
    below are syntactically correct according to the API's signature.
    """
    list_of_inputs = []

    # Input 1: 1D float32 tensor with a name
    input_dict_1 = {
        'value': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'name': 'handle_f32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32 tensor without a name
    input_dict_2 = {
        'value': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar (0D) boolean tensor
    input_dict_3 = {
        'value': np.array(True, dtype=np.bool_),
        'name': 'handle_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D int64 tensor
    input_dict_4 = {
        'value': np.array([-100, 0, 100], dtype=np.int64),
        'name': 'handle_i64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D uint8 tensor without a name
    input_dict_5 = {
        'value': np.zeros((2, 2, 2), dtype=np.uint8),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty tensor with a specific shape
    input_dict_6 = {
        'value': np.empty((0, 5), dtype=np.float32),
        'name': 'handle_empty_shaped'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar int8 tensor
    input_dict_7 = {
        'value': np.array(-128, dtype=np.int8),
        'name': 'handle_i8_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D float64 tensor without a name
    input_dict_8 = {
        'value': np.ones((3, 1), dtype=np.float64),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Another 1D int32 tensor to ensure variety
    input_dict_9 = {
        'value': np.array([9, 8, 7, 6], dtype=np.int32),
        'name': 'handle_another_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A different boolean tensor
    input_dict_10 = {
        'value': np.array([False, False], dtype=np.bool_),
        'name': 'handle_another_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandle"] = get_tf_raw_ops_getsessionhandle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandle'.")

check_valid('tf.raw_ops.GetSessionHandle', generated_inputs['tf.raw_ops.GetSessionHandle'], lib="tf", suffix=0)
