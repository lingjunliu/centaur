
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_io_parse_tensor_inputs():
    """
    Generates a list of valid inputs for tf.io.parse_tensor.
    """
    list_of_inputs = []

    def serialize_to_numpy_object(tensor):
        """Helper to serialize a tensor and wrap it in a 0-D numpy array of dtype=object."""
        serialized_bytes = tf.io.serialize_tensor(tensor).numpy()
        return np.array(serialized_bytes, dtype=object)

    # Input 1: 1D int32 tensor
    tensor1 = tf.constant([10, 20, 30], dtype=tf.int32)
    input_dict1 = {
        'serialized': serialize_to_numpy_object(tensor1),
        'out_type': np.int32,
        'name': 'parse_1d_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float32 tensor
    tensor2 = tf.constant([[1.1, 2.2], [3.3, 4.4]], dtype=tf.float32)
    input_dict2 = {
        'serialized': serialize_to_numpy_object(tensor2),
        'out_type': np.float32,
        'name': 'parse_2d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar (0D) int64 tensor
    tensor3 = tf.constant(9876543210, dtype=tf.int64)
    input_dict3 = {
        'serialized': serialize_to_numpy_object(tensor3),
        'out_type': np.int64,
        'name': 'parse_scalar_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D string tensor
    tensor4 = tf.constant(["hello", "world"], dtype=tf.string)
    input_dict4 = {
        'serialized': serialize_to_numpy_object(tensor4),
        'out_type': np.string_,
        'name': 'parse_1d_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D bool tensor
    tensor5 = tf.constant([True, False, True], dtype=tf.bool)
    input_dict5 = {
        'serialized': serialize_to_numpy_object(tensor5),
        'out_type': np.bool_,
        'name': 'parse_1d_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with negative float values, no name
    tensor6 = tf.constant([[-1.5, -2.5], [-3.5, -4.5]], dtype=tf.float64)
    input_dict6 = {
        'serialized': serialize_to_numpy_object(tensor6),
        'out_type': np.float64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty tensor (shape [0])
    tensor7 = tf.constant([], dtype=tf.float32)
    input_dict7 = {
        'serialized': serialize_to_numpy_object(tensor7),
        'out_type': np.float32,
        'name': 'parse_empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Empty tensor with non-zero dimension (shape [5, 0])
    tensor8 = tf.constant(np.empty((5, 0)), dtype=tf.int16)
    input_dict8 = {
        'serialized': serialize_to_numpy_object(tensor8),
        'out_type': np.int16,
        'name': 'parse_empty_2d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: uint8 tensor
    tensor9 = tf.constant([0, 100, 255], dtype=tf.uint8)
    input_dict9 = {
        'serialized': serialize_to_numpy_object(tensor9),
        'out_type': np.uint8,
        'name': 'parse_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 3D tensor
    tensor10 = tf.constant(np.arange(8).reshape((2, 2, 2)), dtype=tf.int32)
    input_dict10 = {
        'serialized': serialize_to_numpy_object(tensor10),
        'out_type': np.int32,
        'name': 'parse_3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
