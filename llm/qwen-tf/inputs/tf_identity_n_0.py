
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_identity_n_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    b = tf.constant([[2, 4, -6], [5, 7, 9]], dtype=tf.float32)
    input_dict = {
        "input": [a, b],
        "name": "test1"
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=tf.int32)
    input_dict = {
        "input": [a, b],
        "name": "test2"
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid
    a = tf.constant([1, 2, 3, 4], dtype=tf.float64)
    b = tf.constant([5, 6, 7, 8], dtype=tf.float64)
    input_dict = {
        "input": [a, b],
        "name": "test3"
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid
    a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int64)
    b = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int64)
    input_dict = {
        "input": [a, b],
        "name": "test4"
    }
    list_of_inputs.append(input_dict)

    # Input 5, valid
    a = tf.constant([[-1, -2, -3], [-4, -5, -6]], dtype=tf.float32)
    b = tf.constant([[-7, -8, -9], [-10, -11, -12]], dtype=tf.float32)
    input_dict = {
        "input": [a, b],
        "name": "test5"
    }
    list_of_inputs.append(input_dict)

    # Input 6, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=tf.int32)
    input_dict = {
        "input": [a, b],
        "name": "test6"
    }
    list_of_inputs.append(input_dict)

    # Input 7, valid
    a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float64)
    b = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float64)
    input_dict = {
        "input": [a, b],
        "name": "test7"
    }
    list_of_inputs.append(input_dict)

    # Input 8, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=tf.float32)
    input_dict = {
        "input": [a, b],
        "name": "test8"
    }
    list_of_inputs.append(input_dict)

    # Input 9, valid
    a = tf.constant([[-1, -2, -3], [-4, -5, -6]], dtype=tf.int32)
    b = tf.constant([[-7, -8, -9], [-10, -11, -12]], dtype=tf.int32)
    input_dict = {
        "input": [a, b],
        "name": "test9"
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float64)
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=tf.float64)
    input_dict = {
        "input": [a, b],
        "name": "test10"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
