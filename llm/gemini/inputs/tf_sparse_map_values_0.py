
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_map_values_inputs():
    list_of_inputs = []

    # Input 1: Basic example with tf.ones_like
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.ones_like,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tf.multiply with two SparseTensors
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values1 = np.array([1, 2, 3], dtype=np.int32)
    values2 = np.array([4, 5, 6], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s1 = tf.SparseTensor(indices=indices, values=values1, dense_shape=dense_shape)
    s2 = tf.SparseTensor(indices=indices, values=values2, dense_shape=dense_shape)

    input_dict = {
        "op": tf.multiply,
        "*args": [s1, s2],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: tf.add with a SparseTensor and a constant
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.add,
        "*args": [s, tf.constant(5)],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tf.negative
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1, -2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.negative,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: tf.math.abs
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([-1, -2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.math.abs,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tf.square
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.square,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float values with tf.math.sin
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([0.0, np.pi/2, np.pi], dtype=np.float32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.math.sin,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean values with tf.logical_not
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([True, False, True], dtype=np.bool_)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.logical_not,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different dense_shape
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([1, 2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.ones_like,
        "*args": [s],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tf.add with scalar
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2])
    s = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    input_dict = {
        "op": tf.add,
        "*args": [s, tf.constant(2)],
        "**kwargs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.map_values"] = tf_sparse_map_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.map_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.map_values'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.map_values', generated_inputs['tf.sparse.map_values'], lib="tf", suffix=0)
