
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_take_while_inputs():
    list_of_inputs = []

    # Input 1: Simple predicate always true
    def predicate1(x):
        return tf.constant(True)
    input_dict1 = {"predicate": [predicate1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Simple predicate always false
    def predicate2(x):
        return tf.constant(False)
    input_dict2 = {"predicate": [predicate2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Predicate based on value
    def predicate3(x):
        return x < 5
    input_dict3 = {"predicate": [predicate3]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 5: Predicate using tf.reduce_sum
    def predicate5(x):
        return tf.reduce_sum(x) < 10
    input_dict5 = {"predicate": [predicate5]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 7: Predicate that handles potentially empty tensor
    def predicate7(x):
        return tf.cond(tf.size(x) > 0, lambda: x[0] < 5, lambda: tf.constant(False))
    input_dict7 = {"predicate": [predicate7]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Predicate with a more complex condition
    def predicate8(x):
        return tf.logical_and(tf.reduce_sum(x) > 0, tf.reduce_max(x) < 10)
    input_dict8 = {"predicate": [predicate8]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Predicate using tf.cast
    def predicate9(x):
        return tf.cast(tf.reduce_sum(x) > 2, tf.bool)
    input_dict9 = {"predicate": [predicate9]}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Predicate with multiple conditions and different data types.
    def predicate10(x):
        return tf.logical_and(tf.reduce_mean(tf.cast(x, tf.float32)) < 5, tf.reduce_max(x) > 0)
    input_dict10 = {"predicate": [predicate10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Predicate for string tensors. Removing due to potential incompatibility.
    #def predicate11(x):
    #    return tf.strings.length(x) > 3
    #input_dict11 = {"predicate": [predicate11]}
    #list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: More complicated condition.
    def predicate12(x):
      return tf.reduce_all(x > -1)

    input_dict12 = {"predicate": [predicate12]}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    # Input 13: Predicate for equality
    def predicate13(x):
        return tf.equal(x, 5)
    input_dict13 = {"predicate": [predicate13]}
    list_of_inputs.append(copy.deepcopy(input_dict13))

    # Input 14: Another predicate
    def predicate14(x):
        return tf.reduce_any(x > 3)
    input_dict14 = {"predicate": [predicate14]}
    list_of_inputs.append(copy.deepcopy(input_dict14))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.take_while"] = tf_data_experimental_take_while_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.take_while' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.take_while'.")

check_valid('tf.data.experimental.take_while', generated_inputs['tf.data.experimental.take_while'], lib="tf", suffix=0)
