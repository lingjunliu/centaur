
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cast_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    DstT = tf.float32
    Truncate = False
    name = "cast1"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([1.5, 2.7, 3.9], dtype=np.float64)
    DstT = tf.int64
    Truncate = True
    name = "cast2"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    DstT = tf.float16
    Truncate = False
    name = "cast3"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([True, False, True], dtype=np.bool_)
    DstT = tf.int32
    Truncate = True
    name = "cast4"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    DstT = tf.complex64
    Truncate = False
    name = "cast5"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    DstT = tf.int16
    Truncate = True
    name = "cast6"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3], dtype=np.int64)
    DstT = tf.float64
    Truncate = False
    name = "cast7"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.1, 2.2, 3.3], dtype=np.float16)
    DstT = tf.int8
    Truncate = True
    name = "cast8"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    DstT = tf.float32
    Truncate = False
    name = "cast9"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    DstT = tf.float64
    Truncate = True
    name = "cast10"
    input_dict = {"x": x, "DstT": DstT, "Truncate": Truncate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cast"] = tf_raw_ops_cast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cast'.")

check_valid('tf.raw_ops.Cast', generated_inputs['tf.raw_ops.Cast'], lib="tf", suffix=0)
