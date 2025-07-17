
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
    op = tf.math.abs
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.math.square
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.math.add
    args = [s, tf.constant(5)]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dense shape
    s = tf.sparse.from_dense([[1, 2], [0, 4]])
    op = tf.math.negative
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different values
    s = tf.sparse.from_dense([[10, -2, 0], [0, 4, -1]])
    op = tf.math.sin
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using kwargs
    s = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0], [1, 0, 0]])
    op = tf.math.multiply
    args = [s]
    kwargs = {'y': tf.constant(2)}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    s = tf.sparse.from_dense([[[1, 0], [0, 2]], [[3, 0], [0, 4]]])
    op = tf.math.cos
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: multiple sparse tensors with same shape
    s1 = tf.sparse.from_dense([[1, 2, 0], [0, 4, 0]])
    s2 = tf.sparse.from_dense([[5, 0, 1], [2, 0, 3]])
    op = tf.math.add
    args = [s1, s2]
    kwargs = {}

    input_dict = {
        "op": s1,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: with different dtype
    s = tf.sparse.from_dense(np.array([[1.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32))
    op = tf.math.exp
    args = [s]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: with different dtype and op
    s = tf.sparse.from_dense(np.array([[1, 2, 0], [0, 4, 0]], dtype=np.int64))
    op = tf.cast
    args = [s, tf.float32]
    kwargs = {}

    input_dict = {
        "op": s,
        "*args": args,
        "**kwargs": kwargs
    }
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
