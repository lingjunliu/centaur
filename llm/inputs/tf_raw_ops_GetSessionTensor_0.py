
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_get_session_tensor_inputs():
    list_of_inputs = []

    # The tf.raw_ops.GetSessionTensor op requires a valid handle from a
    # populated session state. Since we cannot provide this in a static
    # input generation context, a FailedPreconditionError is expected at runtime.
    # The following inputs are syntactically correct according to the API signature.
    # The 'handle' is provided as np.array(b'some_string', dtype=object)
    # to correctly represent a tf.string scalar.

    # Input 1
    input_dict_1 = {
        'handle': np.array(b'handle_f32', dtype=object),
        'dtype': np.float32,
        'name': 'get_tensor_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'handle': np.array(b'handle_i32', dtype=object),
        'dtype': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'handle': np.array(b'handle_c64', dtype=object),
        'dtype': np.complex64,
        'name': 'get_tensor_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'handle': np.array(b'handle_bool', dtype=object),
        'dtype': np.bool_,
        'name': 'get_tensor_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        'handle': np.array(b'handle_f64', dtype=object),
        'dtype': np.float64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        'handle': np.array(b'handle_i64', dtype=object),
        'dtype': np.int64,
        'name': 'get_tensor_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        'handle': np.array(b'handle_ui8', dtype=object),
        'dtype': np.uint8,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        'handle': np.array(b'handle_str', dtype=object),
        'dtype': np.string_,
        'name': 'get_tensor_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9
    input_dict_9 = {
        'handle': np.array(b'handle_i16', dtype=object),
        'dtype': np.int16,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        'handle': np.array(b'handle_f16', dtype=object),
        'dtype': np.float16,
        'name': 'get_tensor_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11
    input_dict_11 = {
        'handle': np.array(b'handle_c128', dtype=object),
        'dtype': np.complex128,
        'name': 'get_tensor_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionTensor"] = tf_raw_ops_get_session_tensor_inputs()

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
