
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parsetensor_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ParseTensor function.
    """
    list_of_inputs = []

    def create_input_dict(tensor_to_serialize, name):
        """Helper to create a single input dictionary."""
        serialized_tensor_proto = tf.io.serialize_tensor(tensor_to_serialize)
        serialized_bytes = serialized_tensor_proto.numpy()
        # The 'serialized' argument is a scalar string tensor. Its numpy
        # representation is a 0-D array. Using dtype=object ensures a
        # consistent dtype that avoids issues with testing harnesses not
        # recognizing dynamic 'S<n>' dtypes.
        serialized_numpy = np.array(serialized_bytes, dtype=object)

        return {
            'serialized': serialized_numpy,
            'out_type': tensor_to_serialize.dtype.as_numpy_dtype,
            'name': name
        }

    # Input 1: 1D float32 tensor
    tensor_1 = tf.constant([1.0, 2.5, -3.0], dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_1, "float32_1d")))

    # Input 2: 2D int32 tensor with negative values
    tensor_2 = tf.constant([[-1, 2, -3], [4, -5, 6]], dtype=tf.int32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_2, "int32_2d")))

    # Input 3: Scalar bool tensor (True)
    tensor_3 = tf.constant(True, dtype=tf.bool)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_3, "bool_scalar_true")))

    # Input 4: 3D uint8 tensor
    tensor_4 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.uint8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_4, "uint8_3d")))

    # Input 5: 1D string tensor
    tensor_5 = tf.constant(["hello", "world", "tensorflow"], dtype=tf.string)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_5, "string_1d")))

    # Input 6: 2D float64 tensor
    tensor_6 = tf.constant([[1.123456789, 2.987654321]], dtype=tf.float64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_6, "float64_2d")))

    # Input 7: 1D complex64 tensor
    tensor_7 = tf.constant([1+2j, 3-4j, 5+0j], dtype=tf.complex64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_7, "complex64_1d")))

    # Input 8: Empty tensor with shape (0,)
    tensor_8 = tf.constant([], dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_8, "empty_1d_tensor")))

    # Input 9: Scalar int64 tensor
    tensor_9 = tf.constant(9223372036854775807, dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_9, "int64_scalar")))

    # Input 10: 1D complex128 tensor
    tensor_10 = tf.constant([1.1e100 + 2.2e100j, -3.3e100 - 4.4e100j], dtype=tf.complex128)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_10, "complex128_1d")))

    # Input 11: Empty tensor with a non-zero dimension, shape (3, 0)
    tensor_11 = tf.zeros(shape=(3, 0), dtype=tf.int16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_11, "empty_2d_tensor")))

    # Input 12: 4D tensor of ones
    tensor_12 = tf.ones(shape=(1, 2, 1, 3), dtype=tf.float16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_12, "float16_4d")))

    # Input 13: Scalar bool tensor (False)
    tensor_13 = tf.constant(False, dtype=tf.bool)
    list_of_inputs.append(copy.deepcopy(create_input_dict(tensor_13, "bool_scalar_false")))

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
