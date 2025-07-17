
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OnesLike_inputs():
    list_of_inputs = []

    # Input 1: half
    x = np.array([[1, 2], [3, 4]], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16).numpy(), "name": "ones_like_half_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32).numpy(), "name": "ones_like_float32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64).numpy(), "name": "ones_like_float64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int8
    x = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    input_dict = {"x": tf.constant(x, dtype=tf.int8).numpy(), "name": "ones_like_int8_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"x": tf.constant(x, dtype=tf.uint8).numpy(), "name": "ones_like_uint8_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, 3D
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"x": tf.constant(x, dtype=tf.int32).numpy(), "name": "ones_like_int32_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint32
    x = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    input_dict = {"x": tf.constant(x, dtype=tf.uint32).numpy(), "name": "ones_like_uint32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, 1D
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {"x": tf.constant(x, dtype=tf.int64).numpy(), "name": "ones_like_int64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint64
    x = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    input_dict = {"x": tf.constant(x, dtype=tf.uint64).numpy(), "name": "ones_like_uint64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bool
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": tf.constant(x, dtype=tf.bool).numpy(), "name": "ones_like_bool_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_OnesLike_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OnesLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OnesLike'.")

check_valid('tf.raw_ops.OnesLike', generated_inputs['tf.raw_ops.OnesLike'], lib="tf", suffix=0)
