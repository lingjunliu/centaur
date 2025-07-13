
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [tf.constant([['a'], ['b']]), tf.constant([['c'], ['d']])]
    name = "cross_op_1"
    separator = "_X_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.constant([['a', 'b'], ['c', 'd']]), tf.constant([['e', 'f'], ['g', 'h']])]
    name = "cross_op_2"
    separator = "_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.constant([['a']]), tf.constant([['b']]), tf.constant([['c']])]
    name = "cross_op_3"
    separator = "-SEP-"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty inputs
    inputs = [tf.constant([['']]), tf.constant([['']])]
    name = "cross_op_4"
    separator = "*"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer strings
    inputs = [tf.constant([['long_string_1'], ['long_string_2']]), tf.constant([['another_long_string_1'], ['another_long_string_2']])]
    name = "cross_op_5"
    separator = "---"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different rank tensors - Removing since shape mismatch caused error
    # inputs = [tf.constant(['a', 'b']), tf.constant([['c'], ['d']])]
    # name = "cross_op_6"
    # separator = "|||"
    # input_dict = {"inputs": inputs, "name": name, "separator": separator}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes
    inputs = [tf.constant([['a', 'b', 'c'], ['d', 'e', 'f']]), tf.constant([['g'], ['h']])]
    name = "cross_op_7"
    separator = "$$$"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode characters
    inputs = [tf.constant([['你好'], ['世界']]), tf.constant([['TensorFlow'], ['Python']])]
    name = "cross_op_8"
    separator = "~~~"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [tf.constant([['a']]), tf.constant([['b']])]
    name = "cross_op_9"
    separator = "---"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    inputs = [tf.constant([['1'], ['2']]), tf.constant([['3'], ['4']])]
    name = "cross_op_10"
    separator = "..."
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross"] = tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
