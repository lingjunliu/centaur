
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_map_values_inputs():
    list_of_inputs = []

    # Input 1
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.ones_like
    args = [s.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.multiply
    args = [s.values, tf.identity(s.values)]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.add
    args = [s.values, tf.constant(5, dtype=tf.int64)]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    s = tf.sparse.from_dense([[-1, 2, 0], [0, -4, 0], [1, 0, -2]])
    op = tf.abs
    args = [s.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    s = tf.sparse.from_dense([[1.0, 2.0, 0.0], [0.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    op = tf.math.sqrt
    args = [s.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.add
    args = [s.values]
    kwargs = {"y": tf.constant(5, dtype=tf.int64)}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    s1 = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    s2 = tf.sparse.from_dense([[5, 6, 0], [0, 8, 0], [5, 0, 0]])

    op = tf.multiply
    args = [s1.values, s2.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    s = tf.sparse.from_dense([[[1, 0], [0, 2]], [[3, 0], [0, 4]]])
    op = tf.math.log1p
    args = [s.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.cast
    args = [s.values]
    kwargs = {"dtype": tf.float32}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.negative
    args = [s.values]
    kwargs = {}
    input_dict = {"op": op, "*args": args, "**kwargs": kwargs}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.map_values"] = tf_sparse_map_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.map_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.map_values'.")

check_valid('tf.sparse.map_values', generated_inputs['tf.sparse.map_values'], lib="tf", suffix=0)
