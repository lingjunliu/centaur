
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_parse_tensor_inputs():
    list_of_inputs = []

    # Input 1: Simple int32 tensor
    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.int32
    name = "int32_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 tensor
    tensor = tf.constant(np.array([1.0, 2.5, 3.7], dtype=np.float64))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.float64
    name = "float64_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String tensor
    tensor = tf.constant(np.array([b"hello", b"world"], dtype=np.string_))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.string
    name = "string_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    tensor = tf.constant(np.array([True, False, True], dtype=np.bool_))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.bool
    name = "bool_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 tensor
    tensor = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.int32
    name = "2d_int32_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Empty tensor
    tensor = tf.constant(np.array([], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.int32
    name = "empty_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 tensor
    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.int64
    name = "int64_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64
    tensor = tf.constant(np.array([1+1j, 2+2j], dtype=np.complex64))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.complex64
    name = "complex64_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 3 tensor
    tensor = tf.constant(np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int32))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.int32
    name = "rank3_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    tensor = tf.constant(np.array([1, 2, 3], dtype=np.uint8))
    serialized = tf.io.serialize_tensor(tensor).numpy()
    out_type = tf.uint8
    name = "uint8_tensor"

    input_dict = {
        "serialized": serialized,
        "out_type": out_type,
        "name": name
    }
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
