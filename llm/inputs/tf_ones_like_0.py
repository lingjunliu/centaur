
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_like_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.int32, "name": "ones_like_int32", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.float32, "name": "ones_like_float32", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool
    input_tensor = tf.constant([[True, False], [False, True]])
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.bool, "name": "ones_like_bool", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64
    input_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.int64, "name": "ones_like_int64", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    input_tensor = tf.constant([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=tf.complex64)
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.complex64, "name": "ones_like_complex64", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape (3D)
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.int32, "name": "ones_like_3d", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shape (1D)
    input_tensor = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.int32, "name": "ones_like_1d", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    input_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.uint8)
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.uint8, "name": "ones_like_uint8", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
    input_dict = {"input": input_tensor.numpy(), "dtype": tf.float16, "name": "ones_like_float16", "layout": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Layout
    # input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
    # input_dict = {"input": input_tensor.numpy(), "dtype": tf.float16, "name": "ones_like_float16", "layout": tf.constant([1, 2])}
    # list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ones_like"] = tf_ones_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ones_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ones_like'.")

check_valid('tf.ones_like', generated_inputs['tf.ones_like'], lib="tf", suffix=0)
