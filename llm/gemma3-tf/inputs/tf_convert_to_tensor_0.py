
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_convert_to_tensor_inputs():
    list_of_inputs = []

    value1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dtype1 = None
    dtype_hint1 = tf.float32
    name1 = "tensor1"
    input_dict1 = {
        "value": value1,
        "dtype": dtype1,
        "dtype_hint": dtype_hint1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    value2 = np.array([1, 2, 3], dtype=np.int32)
    dtype2 = None
    dtype_hint2 = tf.int64
    name2 = "tensor2"
    input_dict2 = {
        "value": value2,
        "dtype": dtype2,
        "dtype_hint": dtype_hint2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    value3 = np.array([[-1, 2], [3, -4]], dtype=np.float64)
    dtype3 = None
    dtype_hint3 = tf.float32
    name3 = "tensor3"
    input_dict3 = {
        "value": value3,
        "dtype": dtype3,
        "dtype_hint": dtype_hint3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    value4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    dtype4 = None
    dtype_hint4 = tf.int64
    name4 = "tensor4"
    input_dict4 = {
        "value": value4,
        "dtype": dtype4,
        "dtype_hint": dtype_hint4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    value5 = np.array([], dtype=np.float32)
    dtype5 = tf.float32
    dtype_hint5 = None
    name5 = "tensor5"
    input_dict5 = {
        "value": value5,
        "dtype": dtype5,
        "dtype_hint": dtype_hint5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    value6 = np.array([1.0], dtype=np.float64)
    dtype6 = None
    dtype_hint6 = tf.float64
    name6 = "tensor6"
    input_dict6 = {
        "value": value6,
        "dtype": dtype6,
        "dtype_hint": dtype_hint6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.convert_to_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.convert_to_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.convert_to_tensor', generated_inputs['tf.convert_to_tensor'], lib="tf", suffix=0)
