
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_io_parse_tensor_inputs():
    """
    Generates a list of valid inputs for the tf.io.parse_tensor function.
    """
    list_of_inputs = []

    # Case 1: 1D int32 tensor
    tensor_to_serialize_1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    input_dict_1 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_1).numpy(), dtype=object),
        'out_type': tensor_to_serialize_1.dtype.as_numpy_dtype,
        'name': 'parse_int32_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D float32 tensor with negative values
    tensor_to_serialize_2 = tf.constant([[-1.1, 2.2], [-3.3, 4.4]], dtype=tf.float32)
    input_dict_2 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_2).numpy(), dtype=object),
        'out_type': tensor_to_serialize_2.dtype.as_numpy_dtype,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Scalar bool tensor
    tensor_to_serialize_3 = tf.constant(True, dtype=tf.bool)
    input_dict_3 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_3).numpy(), dtype=object),
        'out_type': tensor_to_serialize_3.dtype.as_numpy_dtype,
        'name': 'parse_bool_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 3D uint8 tensor
    tensor_to_serialize_4 = tf.constant(np.arange(24, dtype=np.uint8).reshape(2, 3, 4), dtype=tf.uint8)
    input_dict_4 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_4).numpy(), dtype=object),
        'out_type': tensor_to_serialize_4.dtype.as_numpy_dtype,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 1D string tensor
    tensor_to_serialize_5 = tf.constant(["hello", "world", "tensorflow"], dtype=tf.string)
    input_dict_5 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_5).numpy(), dtype=object),
        'out_type': tensor_to_serialize_5.dtype.as_numpy_dtype,
        'name': 'parse_string_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 2D complex64 tensor
    tensor_to_serialize_6 = tf.constant([[1+2j, 3-4j], [-5+6j, 7+8j]], dtype=tf.complex64)
    input_dict_6 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_6).numpy(), dtype=object),
        'out_type': tensor_to_serialize_6.dtype.as_numpy_dtype,
        'name': 'parse_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Empty float64 tensor
    tensor_to_serialize_7 = tf.constant([], dtype=tf.float64)
    input_dict_7 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_7).numpy(), dtype=object),
        'out_type': tensor_to_serialize_7.dtype.as_numpy_dtype,
        'name': 'parse_empty_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: 4D int16 tensor
    tensor_to_serialize_8 = tf.constant(np.ones((1, 2, 2, 3), dtype=np.int16) * -10, dtype=tf.int16)
    input_dict_8 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_8).numpy(), dtype=object),
        'out_type': tensor_to_serialize_8.dtype.as_numpy_dtype,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: 1D int64 tensor with large numbers
    tensor_to_serialize_9 = tf.constant([10000000000, -20000000000], dtype=tf.int64)
    input_dict_9 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_9).numpy(), dtype=object),
        'out_type': tensor_to_serialize_9.dtype.as_numpy_dtype,
        'name': 'parse_int64_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Case 10: 2D float16 tensor
    tensor_to_serialize_10 = tf.constant([[0.5, -0.25], [1.0, -1.5]], dtype=tf.float16)
    input_dict_10 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_10).numpy(), dtype=object),
        'out_type': tensor_to_serialize_10.dtype.as_numpy_dtype,
        'name': 'parse_float16_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Scalar complex128 tensor
    tensor_to_serialize_11 = tf.constant(1.23456789 + 9.87654321j, dtype=tf.complex128)
    input_dict_11 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_11).numpy(), dtype=object),
        'out_type': tensor_to_serialize_11.dtype.as_numpy_dtype,
        'name': 'parse_complex128_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Case 12: Zero-sized tensor (shape (2,0))
    tensor_to_serialize_12 = tf.zeros((2, 0), dtype=tf.uint32)
    input_dict_12 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_to_serialize_12).numpy(), dtype=object),
        'out_type': tensor_to_serialize_12.dtype.as_numpy_dtype,
        'name': 'parse_zero_sized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.io.parse_tensor"] = get_tf_io_parse_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.parse_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.parse_tensor'.")

check_valid('tf.io.parse_tensor', generated_inputs['tf.io.parse_tensor'], lib="tf", suffix=0)
