
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_parse_tensor_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32
    tensor_1 = tf.constant(3.14, dtype=tf.float32)
    input_dict_1 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_1).numpy(), dtype=object),
        'out_type': tensor_1.dtype.as_numpy_dtype,
        'name': 'scalar_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D int32 vector with negative values
    tensor_2 = tf.constant([1, -2, 3, 0], dtype=tf.int32)
    input_dict_2 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_2).numpy(), dtype=object),
        'out_type': tensor_2.dtype.as_numpy_dtype,
        'name': '1d_int32_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D float64 matrix
    tensor_3 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    input_dict_3 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_3).numpy(), dtype=object),
        'out_type': tensor_3.dtype.as_numpy_dtype,
        'name': 'matrix_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D int64 tensor
    tensor_4 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int64)
    input_dict_4 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_4).numpy(), dtype=object),
        'out_type': tensor_4.dtype.as_numpy_dtype,
        'name': '3d_tensor_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean vector
    tensor_5 = tf.constant([True, False, True], dtype=tf.bool)
    input_dict_5 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_5).numpy(), dtype=object),
        'out_type': tensor_5.dtype.as_numpy_dtype,
        'name': 'bool_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex64 vector
    tensor_6 = tf.constant([1+2j, 3-4j], dtype=tf.complex64)
    input_dict_6 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_6).numpy(), dtype=object),
        'out_type': tensor_6.dtype.as_numpy_dtype,
        'name': 'complex64_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex128 scalar
    tensor_7 = tf.constant(1.5+2.5j, dtype=tf.complex128)
    input_dict_7 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_7).numpy(), dtype=object),
        'out_type': tensor_7.dtype.as_numpy_dtype,
        'name': 'complex128_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty tensor
    tensor_8 = tf.constant([], dtype=tf.float32)
    input_dict_8 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_8).numpy(), dtype=object),
        'out_type': tensor_8.dtype.as_numpy_dtype,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: String tensor
    tensor_9 = tf.constant([b"hello", b"world", b""], dtype=tf.string)
    input_dict_9 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_9).numpy(), dtype=object),
        'out_type': tensor_9.dtype.as_numpy_dtype,
        'name': 'string_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Unsigned integer tensor (uint8)
    tensor_10 = tf.constant([0, 128, 255], dtype=tf.uint8)
    input_dict_10 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_10).numpy(), dtype=object),
        'out_type': tensor_10.dtype.as_numpy_dtype,
        'name': 'uint8_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: int16 tensor with non-standard shape
    tensor_11 = tf.constant([[[1], [2]], [[3], [4]]], shape=(2, 2, 1), dtype=tf.int16)
    input_dict_11 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_11).numpy(), dtype=object),
        'out_type': tensor_11.dtype.as_numpy_dtype,
        'name': 'int16_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Scalar string
    tensor_12 = tf.constant(b"a single string", dtype=tf.string)
    input_dict_12 = {
        'serialized': np.array(tf.io.serialize_tensor(tensor_12).numpy(), dtype=object),
        'out_type': tensor_12.dtype.as_numpy_dtype,
        'name': 'scalar_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.io.parse_tensor"] = tf_io_parse_tensor_inputs()

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
