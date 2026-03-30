
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_take_while_inputs():
    list_of_inputs = []

    # Input 1: Simple predicate returning True
    def predicate1(x):
        return tf.constant(True)
    input_dict1 = {"predicate": [predicate1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Simple predicate with a tensor
    def predicate2(x):
        return x < tf.constant(5, dtype=tf.int32)
    input_dict2 = {"predicate": [predicate2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: More complex predicate with a tensor
    def predicate3(x):
        return tf.logical_and(x > tf.constant(2, dtype=tf.int32), x < tf.constant(8, dtype=tf.int32))
    input_dict3 = {"predicate": [predicate3]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Predicate using tf.reduce_sum with a tensor
    def predicate4(x):
        return tf.reduce_sum(x) < tf.constant(10, dtype=tf.int32)
    input_dict4 = {"predicate": [predicate4]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Predicate with a multi-dimensional tensor
    def predicate5(x):
        return tf.reduce_sum(x) < tf.constant(20, dtype=tf.int32)
    input_dict5 = {"predicate": [predicate5]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Predicate with logical_or
    def predicate6(x):
        return tf.logical_or(tf.reduce_sum(x) > tf.constant(15, dtype=tf.int32), x[0] < tf.constant(3, dtype=tf.int32))
    input_dict6 = {"predicate": [predicate6]}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Predicate using abs
    def predicate7(x):
        return tf.math.abs(tf.reduce_sum(x)) < tf.constant(10, dtype=tf.int32)
    input_dict7 = {"predicate": [predicate7]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Predicate with negative values
    def predicate8(x):
        return tf.reduce_sum(x) > tf.constant(-5, dtype=tf.int32)
    input_dict8 = {"predicate": [predicate8]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Predicate using size
    def predicate9(x):
        return tf.size(x) < tf.constant(10, dtype=tf.int32)
    input_dict9 = {"predicate": [predicate9]}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Predicate using reduce_mean
    def predicate10(x):
        return tf.reduce_mean(tf.cast(x, tf.float32)) < tf.constant(5, dtype=tf.float32)
    input_dict10 = {"predicate": [predicate10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.take_while"] = tf_data_experimental_take_while_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.take_while' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.take_while'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.take_while', generated_inputs['tf.data.experimental.take_while'], lib="tf", suffix=0)
