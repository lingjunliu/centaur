
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_cross_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [tf.constant([['a'], ['b']]), tf.constant([['c'], ['d']])]
    name = "cross_1"
    separator = "_X_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=['a', 'b'], dense_shape=[2, 1]), tf.constant([['c'], ['d']])]
    name = "cross_2"
    separator = "_Y_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=['a', 'b'], dense_shape=[2, 1]), tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=['c', 'd'], dense_shape=[2, 1])]
    name = "cross_3"
    separator = "_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.constant([['a', 'b'], ['c', 'd']]), tf.constant([['e', 'f'], ['g', 'h']])]
    name = "cross_4"
    separator = ""
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [tf.constant([['a']]), tf.constant([['b']]), tf.constant([['c']])]
    name = "cross_5"
    separator = "++"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [tf.constant([['a', 'b', 'c']]), tf.constant([['d', 'e', 'f']])]
    name = "cross_6"
    separator = "--"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [tf.sparse.SparseTensor(indices=[[0, 0]], values=['a'], dense_shape=[1, 1]), tf.constant([['b']])]
    name = "cross_7"
    separator = "..."
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [tf.constant([['a'], ['b'], ['c']]), tf.constant([['d'], ['e'], ['f']])]
    name = "cross_8"
    separator = "_!_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=['a', 'b'], dense_shape=[2, 2]), tf.constant([['c'], ['d']])]
    name = "cross_9"
    separator = "_@_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = [tf.constant([['a'], ['b']]), tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=['c', 'd'], dense_shape=[2, 1])]
    name = "cross_10"
    separator = "_#_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross"] = tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
