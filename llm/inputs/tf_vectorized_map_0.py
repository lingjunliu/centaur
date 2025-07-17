
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_vectorized_map_inputs():
    list_of_inputs = []

    # Input 1: Simple addition
    fn1 = lambda x: x + 1

    elems1 = np.array([1, 2, 3], dtype=np.float32)
    fallback_to_while_loop1 = True
    warn1 = True

    input_dict1 = {
        "fn": [fn1],
        "elems": elems1,
        "fallback_to_while_loop": fallback_to_while_loop1,
        "warn": warn1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Squaring
    fn2 = lambda x: x * x

    elems2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fallback_to_while_loop2 = False
    warn2 = False

    input_dict2 = {
        "fn": [fn2],
        "elems": elems2,
        "fallback_to_while_loop": fallback_to_while_loop2,
        "warn": warn2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Matrix multiplication
    fn3 = lambda x: tf.matmul(x, x, transpose_b=True)

    elems3 = np.random.rand(3, 2, 2).astype(np.float32)
    fallback_to_while_loop3 = True
    warn3 = False

    input_dict3 = {
        "fn": [fn3],
        "elems": elems3,
        "fallback_to_while_loop": fallback_to_while_loop3,
        "warn": warn3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: More complex function
    fn4 = lambda x: tf.sin(x) + tf.cos(x)

    elems4 = np.linspace(0, 2*np.pi, 5).astype(np.float32)
    fallback_to_while_loop4 = False
    warn4 = True
    input_dict4 = {
        "fn": [fn4],
        "elems": elems4,
        "fallback_to_while_loop": fallback_to_while_loop4,
        "warn": warn4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multi-dimensional array
    fn5 = lambda x: x * 2.0

    elems5 = np.random.rand(2, 3, 4).astype(np.float32)
    fallback_to_while_loop5 = True
    warn5 = True
    input_dict5 = {
        "fn": [fn5],
        "elems": elems5,
        "fallback_to_while_loop": fallback_to_while_loop5,
        "warn": warn5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Using negative values
    fn6 = lambda x: x * -1.0

    elems6 = np.array([-1, -2, -3, 4, 5], dtype=np.float32)
    fallback_to_while_loop6 = False
    warn6 = False
    input_dict6 = {
        "fn": [fn6],
        "elems": elems6,
        "fallback_to_while_loop": fallback_to_while_loop6,
        "warn": warn6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Zero values
    fn7 = lambda x: x + 0.0

    elems7 = np.array([0, 0, 0], dtype=np.float32)
    fallback_to_while_loop7 = True
    warn7 = False
    input_dict7 = {
        "fn": [fn7],
        "elems": elems7,
        "fallback_to_while_loop": fallback_to_while_loop7,
        "warn": warn7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Broadcasting with single element
    fn8 = lambda x: x + 1.0

    elems8 = np.array([1.0], dtype=np.float32)
    fallback_to_while_loop8 = False
    warn8 = True
    input_dict8 = {
        "fn": [fn8],
        "elems": elems8,
        "fallback_to_while_loop": fallback_to_while_loop8,
        "warn": warn8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Complex function with different dtype
    fn9 = lambda x: tf.cast(x, tf.float64) * 2

    elems9 = np.array([1, 2, 3], dtype=np.int32)
    fallback_to_while_loop9 = True
    warn9 = True
    input_dict9 = {
        "fn": [fn9],
        "elems": elems9,
        "fallback_to_while_loop": fallback_to_while_loop9,
        "warn": warn9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Identity function
    fn10 = lambda x: x

    elems10 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    fallback_to_while_loop10 = False
    warn10 = False
    input_dict10 = {
        "fn": [fn10],
        "elems": elems10,
        "fallback_to_while_loop": fallback_to_while_loop10,
        "warn": warn10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.vectorized_map"] = tf_vectorized_map_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.vectorized_map' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.vectorized_map'.")

check_valid('tf.vectorized_map', generated_inputs['tf.vectorized_map'], lib="tf", suffix=0)
