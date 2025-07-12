
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

    # Input 3: Predicate based on element value (less than 5)
    def predicate3(x):
        return tf.less(x, tf.constant(5, dtype=x.dtype))
    input_dict3 = {"predicate": [predicate3]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Predicate based on element value (greater than 10)
    def predicate4(x):
        return tf.greater(x, tf.constant(10, dtype=x.dtype))
    input_dict4 = {"predicate": [predicate4]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Predicate with multiple conditions
    def predicate5(x):
        return tf.logical_and(tf.greater(x, tf.constant(2, dtype=x.dtype)), tf.less(x, tf.constant(8, dtype=x.dtype)))
    input_dict5 = {"predicate": [predicate5]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Predicate with a tensor input (checks if all elements are positive)
    def predicate6(x):
        return tf.reduce_all(tf.greater(x, tf.constant(0, dtype=x.dtype)))
    input_dict6 = {"predicate": [predicate6]}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Predicate with a tensor input (checks if any element is greater than 5)
    def predicate7(x):
        return tf.reduce_any(tf.greater(x, tf.constant(5, dtype=x.dtype)))
    input_dict7 = {"predicate": [predicate7]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Predicate with complex boolean logic.
    def predicate8(x):
      return tf.logical_or(tf.reduce_all(tf.greater(x, tf.constant(0, dtype=x.dtype))), tf.reduce_any(tf.less(x, tf.constant(-2, dtype=x.dtype))))
    input_dict8 = {"predicate": [predicate8]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Predicate using tf.math functions
    def predicate9(x):
        return tf.math.reduce_sum(x) < tf.constant(20, dtype=x.dtype)
    input_dict9 = {"predicate": [predicate9]}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
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
