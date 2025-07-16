
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_debug_gradient_identity_inputs():
    list_of_inputs = []

    # Input 1: Float32 tensor, no name
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 tensor, with name
    input_tensor = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "debug_identity_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 tensor, 2D array
    input_tensor = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64 tensor, 2D array, name specified
    input_tensor = np.array([[7, 8], [9, 10]], dtype=np.int64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "debug_identity_int64_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bool tensor
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor (byte strings)
    input_tensor = np.array([b"hello", b"world"], dtype=np.object_)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "debug_identity_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 tensor
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 tensor, 2D array
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "debug_identity_complex128_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int8 tensor with negative values
    input_tensor = np.array([-1, 0, 1], dtype=np.int8)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Uint8 tensor
    input_tensor = np.array([1, 2, 3], dtype=np.uint8)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "debug_identity_uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DebugGradientIdentity"] = tf_raw_ops_debug_gradient_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DebugGradientIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientIdentity'.")

check_valid('tf.raw_ops.DebugGradientIdentity', generated_inputs['tf.raw_ops.DebugGradientIdentity'], lib="tf", suffix=0)
