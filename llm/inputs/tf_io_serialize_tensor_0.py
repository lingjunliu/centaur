
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    # Input 1: Scalar integer tensor
    tensor = tf.constant(10)
    name = "scalar_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float tensor
    tensor = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    name = "1d_float"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int tensor
    tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32)
    name = "2d_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean tensor
    tensor = tf.constant([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=tf.bool)
    name = "3d_bool"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty tensor
    tensor = tf.constant([], dtype=tf.int32)
    name = "empty"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_tensor'.")

check_valid('tf.io.serialize_tensor', generated_inputs['tf.io.serialize_tensor'], lib="tf", suffix=0)
