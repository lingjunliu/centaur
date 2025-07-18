
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_get_session_handle_v2_inputs():
    list_of_inputs = []

    # The error "GetSessionHandle called on null session state" is not caused by
    # invalid input values but by the execution context. This op requires a
    # TensorFlow v1-style session to be active, which is not the case in the
    # default eager execution environment of TF2. The provided inputs are
    # valid according to the function's signature.

    # Input 1: Scalar float32
    input_dict = {
        'value': np.array(3.14, dtype=np.float32),
        'name': 'handle_float_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar int32
    input_dict = {
        'value': np.array(42, dtype=np.int32),
        'name': 'handle_int_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 vector
    input_dict = {
        'value': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'name': 'handle_float_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 vector
    input_dict = {
        'value': np.array([-1, 0, 1], dtype=np.int32),
        'name': 'handle_int_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 matrix
    input_dict = {
        'value': np.ones((2, 3), dtype=np.float64),
        'name': 'handle_float64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int64 matrix
    input_dict = {
        'value': np.arange(6, dtype=np.int64).reshape(2, 3),
        'name': 'handle_int64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 tensor
    input_dict = {
        'value': np.random.randn(2, 2, 2).astype(np.float32),
        'name': 'handle_float_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 tensor of zeros
    input_dict = {
        'value': np.zeros((3, 2, 1), dtype=np.int32),
        'name': 'handle_int_3d_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor with non-zero rank
    input_dict = {
        'value': np.empty((5, 0), dtype=np.float32),
        'name': 'handle_empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean tensor
    input_dict = {
        'value': np.array([True, False, True], dtype=np.bool_),
        'name': 'handle_bool_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandleV2"] = get_tf_raw_ops_get_session_handle_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandleV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandleV2'.")

check_valid('tf.raw_ops.GetSessionHandleV2', generated_inputs['tf.raw_ops.GetSessionHandleV2'], lib="tf", suffix=0)
