
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parsetensor_inputs():
    list_of_inputs = []

    # Helper to create inputs
    def create_input(tf_tensor, name):
        proto = tf.make_tensor_proto(tf_tensor)
        serialized_proto = proto.SerializeToString()
        return {
            'serialized': np.array(serialized_proto, dtype=object),
            'out_type': tf_tensor.dtype.as_numpy_dtype,
            'name': name
        }

    # Input 1: 1D float32 tensor
    tf_tensor_1 = tf.constant([1.1, -2.2, 3.3, 0.0], dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_1, 'parse_float32_vector')))

    # Input 2: 2D int64 tensor with no name
    tf_tensor_2 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_2, None)))

    # Input 3: Scalar (0D) boolean tensor
    tf_tensor_3 = tf.constant(True, dtype=tf.bool)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_3, 'parse_bool_scalar')))

    # Input 4: 3D float64 tensor
    tf_tensor_4 = tf.constant([[[1.0], [0.0]], [[-3.5], [4.1]]], dtype=tf.float64)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_4, 'parse_float64_3d')))

    # Input 5: Empty 1D int32 tensor
    tf_tensor_5 = tf.constant([], dtype=tf.int32)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_5, 'parse_empty_tensor')))

    # Input 6: 1D complex64 tensor
    tf_tensor_6 = tf.constant([1+2j, -3-4j, 5j], dtype=tf.complex64)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_6, 'parse_complex64_vector')))

    # Input 7: 2D complex128 tensor
    tf_tensor_7 = tf.constant([[1.123+2.456j], [-3.789-4.012j]], dtype=tf.complex128)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_7, 'parse_complex128_matrix')))

    # Input 8: Scalar uint8 tensor
    tf_tensor_8 = tf.constant(255, dtype=tf.uint8)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_8, 'parse_uint8_scalar')))

    # Input 9: 2D int16 tensor with boundary values
    tf_tensor_9 = tf.constant([[-32768, 32767], [0, -1]], dtype=tf.int16)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_9, 'parse_int16_matrix')))

    # Input 10: 4D tensor of zeros
    tf_tensor_10 = tf.constant(np.zeros((1, 2, 1, 2)), dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_10, 'parse_4d_zeros_tensor')))

    # Input 11: Empty tensor with a specific shape (1, 0, 2)
    tf_tensor_11 = tf.zeros(shape=(1, 0, 2), dtype=tf.int32)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_11, 'parse_empty_tensor_with_shape')))
    
    # Input 12: Scalar int8 tensor
    tf_tensor_12 = tf.constant(-128, dtype=tf.int8)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_12, 'parse_int8_scalar')))
    
    # Input 13: Scalar uint16 tensor
    tf_tensor_13 = tf.constant(65535, dtype=tf.uint16)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_13, 'parse_uint16_scalar')))
    
    # Input 14: Large 1D uint32 tensor
    tf_tensor_14 = tf.constant(np.arange(100, dtype=np.uint32), dtype=tf.uint32)
    list_of_inputs.append(copy.deepcopy(create_input(tf_tensor_14, 'parse_large_uint32_vector')))

    return list_of_inputs

generated_inputs["tf.raw_ops.ParseTensor"] = tf_raw_ops_parsetensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ParseTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParseTensor'.")

check_valid('tf.raw_ops.ParseTensor', generated_inputs['tf.raw_ops.ParseTensor'], lib="tf", suffix=0)
