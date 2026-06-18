
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    # Input 1: 0D int32 scalar
    input_dict = {
        "tensor": np.array(42, dtype=np.int32),
        "name": "scalar_int"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values
    input_dict = {
        "tensor": np.array([-1.2, 0.0, 3.14], dtype=np.float32),
        "name": "float_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    input_dict = {
        "tensor": np.array([[True, False], [False, True]], dtype=bool),
        "name": "bool_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array
    input_dict = {
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64),
        "name": "int64_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array with special values
    input_dict = {
        "tensor": np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
        "name": "float64_specials"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint8 array
    input_dict = {
        "tensor": np.array([[0, 255], [128, 64]], dtype=np.uint8),
        "name": "uint8_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    input_dict = {
        "tensor": np.random.uniform(-1, 1, (2, 3, 3, 3)).astype(np.float32),
        "name": "float_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array (containing zeros and ones)
    input_dict = {
        "tensor": np.ones((2, 2, 2), dtype=np.int32),
        "name": "int32_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D float32 array
    input_dict = {
        "tensor": np.array([], dtype=np.float32),
        "name": "empty_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array with large values
    input_dict = {
        "tensor": np.array([9223372036854775807, -9223372036854775808, 0], dtype=np.int64),
        "name": "int64_large"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.serialize_tensor', generated_inputs['tf.io.serialize_tensor'], lib="tf", suffix=0)
