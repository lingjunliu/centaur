
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_vectorized_map_inputs():
    list_of_inputs = []

    # Input 1
    def fn1(x):
        return x * 2
    elems1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    fallback_to_while_loop1 = True
    warn1 = True
    input_dict1 = {"fn": fn1, "elems": tf.convert_to_tensor(elems1), "fallback_to_while_loop": fallback_to_while_loop1, "warn": warn1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    def fn2(x):
        return x + 1
    elems2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    fallback_to_while_loop2 = False
    warn2 = False
    input_dict2 = {"fn": fn2, "elems": tf.convert_to_tensor(elems2), "fallback_to_while_loop": fallback_to_while_loop2, "warn": warn2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    def fn3(x):
        return tf.reduce_sum(x)
    elems3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    fallback_to_while_loop3 = True
    warn3 = False
    input_dict3 = {"fn": fn3, "elems": tf.convert_to_tensor(elems3), "fallback_to_while_loop": fallback_to_while_loop3, "warn": warn3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    def fn4(x):
        return x**2
    elems4 = np.array([-1, 0, 1], dtype=np.float32)
    fallback_to_while_loop4 = False
    warn4 = True
    input_dict4 = {"fn": fn4, "elems": tf.convert_to_tensor(elems4), "fallback_to_while_loop": fallback_to_while_loop4, "warn": warn4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    def fn5(x):
        return tf.constant([1, 2, 3], dtype=tf.float32)
    elems5 = np.array([1, 2], dtype=np.float32)
    fallback_to_while_loop5 = True
    warn5 = True
    input_dict5 = {"fn": fn5, "elems": tf.convert_to_tensor(elems5), "fallback_to_while_loop": fallback_to_while_loop5, "warn": warn5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    def fn6(x):
      return tf.reshape(x, [1, -1])
    elems6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    fallback_to_while_loop6 = False
    warn6 = False
    input_dict6 = {"fn": fn6, "elems": tf.convert_to_tensor(elems6), "fallback_to_while_loop": fallback_to_while_loop6, "warn": warn6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    def fn7(x):
        return x
    elems7 = np.array([[[1],[2]],[[3],[4]]], dtype=np.float32)
    fallback_to_while_loop7 = True
    warn7 = True
    input_dict7 = {"fn": fn7, "elems": tf.convert_to_tensor(elems7), "fallback_to_while_loop": fallback_to_while_loop7, "warn": warn7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    def fn8(x):
        return x + 5
    elems8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fallback_to_while_loop8 = False
    warn8 = True
    input_dict8 = {"fn": fn8, "elems": tf.convert_to_tensor(elems8), "fallback_to_while_loop": fallback_to_while_loop8, "warn": warn8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    def fn9(x):
        return tf.cast(x, tf.int32)
    elems9 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    fallback_to_while_loop9 = True
    warn9 = False
    input_dict9 = {"fn": fn9, "elems": tf.convert_to_tensor(elems9), "fallback_to_while_loop": fallback_to_while_loop9, "warn": warn9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    def fn10(x):
        return tf.zeros_like(x)
    elems10 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    fallback_to_while_loop10 = False
    warn10 = False
    input_dict10 = {"fn": fn10, "elems": tf.convert_to_tensor(elems10), "fallback_to_while_loop": fallback_to_while_loop10, "warn": warn10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11 - Example with different dtype
    def fn11(x):
        return x + 1
    elems11 = np.array([1, 2, 3], dtype=np.int32)
    fallback_to_while_loop11 = True
    warn11 = True
    input_dict11 = {"fn": fn11, "elems": tf.convert_to_tensor(elems11, dtype=tf.int32), "fallback_to_while_loop": fallback_to_while_loop11, "warn": warn11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12 - Example with boolean dtype
    def fn12(x):
        return tf.logical_not(x)
    elems12 = np.array([True, False, True], dtype=np.bool_)
    fallback_to_while_loop12 = False
    warn12 = False
    input_dict12 = {"fn": fn12, "elems": tf.convert_to_tensor(elems12), "fallback_to_while_loop": fallback_to_while_loop12, "warn": warn12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.vectorized_map"] = tf_vectorized_map_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.vectorized_map' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.vectorized_map'.")

check_valid('tf.vectorized_map', generated_inputs['tf.vectorized_map'], lib="tf", suffix=0)
