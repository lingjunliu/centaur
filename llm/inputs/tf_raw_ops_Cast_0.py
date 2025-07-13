
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cast_inputs():
    list_of_inputs = []

    # Input 1: float32 to int32, no truncation
    x = np.array([1.5, 2.7, 3.9], dtype=np.float32)
    DstT = tf.int32
    Truncate = False
    name = "cast_float_to_int"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2: int64 to float64, no truncation
    x = np.array([-10, 0, 10000000000], dtype=np.int64)
    DstT = tf.float64
    Truncate = False
    name = "cast_int_to_float"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3: complex64 to float32, with truncation
    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    DstT = tf.float32
    Truncate = True
    name = "cast_complex_to_float_truncate"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4: float64 to int32, with truncation
    x = np.array([1.9, -2.5, 3.1], dtype=np.float64)
    DstT = tf.int32
    Truncate = True
    name = "cast_float_to_int_truncate"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5: int32 to int64, no truncation, multi-dimensional
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    DstT = tf.int64
    Truncate = False
    name = "cast_int_to_int_multidimensional"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6: bool to float32
    x = np.array([True, False, True], dtype=np.bool_)
    DstT = tf.float32
    Truncate = False
    name = "cast_bool_to_float"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7: float32 to bool
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    DstT = tf.bool
    Truncate = False
    name = "cast_float_to_bool"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8: string to string
    x = np.array(["hello", "world"], dtype=np.string_)
    DstT = tf.string
    Truncate = False
    name = "cast_string_to_string"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)
    
    # Input 9: uint8 to int32
    x = np.array([255, 0, 128], dtype=np.uint8)
    DstT = tf.int32
    Truncate = False
    name = "cast_uint8_to_int32"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)
    
    # Input 10: int8 to float32
    x = np.array([-128, 0, 127], dtype=np.int8)
    DstT = tf.float32
    Truncate = False
    name = "cast_int8_to_float32"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)
    
    # Input 11: float16 to float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    DstT = tf.float32
    Truncate = False
    name = "cast_float16_to_float32"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_cast_inputs()
generated_inputs["tf.raw_ops.Cast"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cast'.")

check_valid('tf.raw_ops.Cast', generated_inputs['tf.raw_ops.Cast'], lib="tf", suffix=0)
