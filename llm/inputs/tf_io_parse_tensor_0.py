
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_parse_tensor_inputs():
    list_of_inputs = []

    # Input 1
    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int32, "name": "test_tensor_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.float32, "name": "test_tensor_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int64))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int64, "name": "test_tensor_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.float64, "name": "test_tensor_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = tf.constant(np.array([True, False, True], dtype=np.bool_))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.bool_, "name": "test_tensor_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = tf.constant(["a", "b", "c"], dtype=tf.string)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.dtype('S'), "name": "test_tensor_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Multi-dimensional int
    tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int16, "name": "test_tensor_int16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Negative Values
    tensor = tf.constant(np.array([-1, -2, -3], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int32, "name": "test_tensor_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Empty Array
    tensor = tf.constant(np.array([], dtype=np.float32))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.float32, "name": "test_tensor_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Different name
    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int8))
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int8, "name": "another_test_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.parse_tensor"] = tf_io_parse_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.parse_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.parse_tensor'.")

check_valid('tf.io.parse_tensor', generated_inputs['tf.io.parse_tensor'], lib="tf", suffix=0)
