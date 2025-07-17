
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_add_inputs():
    list_of_inputs = []

    # Input 1: Sparse + Dense, threshold 0
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.constant([[3, 0, 1], [0, 4, 0]], dtype=tf.int32)
    threshold = tf.constant(0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Dense + Sparse, threshold 0
    a = tf.constant([[3, 0, 1], [0, 4, 0]], dtype=tf.int32)
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    threshold = tf.constant(0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sparse + Sparse, threshold > 0
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[-1.0, 1.0], dense_shape=[2, 3])
    threshold = tf.constant(0.5, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sparse + Sparse, threshold = 0, different values
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0]], values=[3, 4], dense_shape=[2, 3])
    threshold = tf.constant(0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sparse + Sparse, all values cancel out
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[-1.0, -2.0], dense_shape=[2, 3])
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sparse + Dense, different dtype
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.constant([[3.0, 0.0, 1.0], [0.0, 4.0, 0.0]], dtype=tf.float32)
    threshold = tf.constant(0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dense + Sparse, different dtype
    a = tf.constant([[3.0, 0.0, 1.0], [0.0, 4.0, 0.0]], dtype=tf.float32)
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    threshold = tf.constant(0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs
generated_inputs = {}
generated_inputs["tf.sparse.add"] = tf_sparse_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.add'.")

check_valid('tf.sparse.add', generated_inputs['tf.sparse.add'], lib="tf", suffix=0)
