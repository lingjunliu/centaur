
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_fill_empty_rows_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([5, 6])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0)
    name = "sparse_fill_1"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [1, 2], [3, 3]])
    values = np.array([1, 2, 3])
    shape = np.array([4, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0)
    name = "sparse_fill_2"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 1], [0, 2]])
    values = np.array([1.0, 2.0, 3.0])
    shape = np.array([1, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0)
    name = "sparse_fill_3"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[2, 1]])
    values = np.array([5])
    shape = np.array([5, 5])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(-1)
    name = "sparse_fill_4"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1, 0, 1])
    shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0)
    name = "sparse_fill_6"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 5], [1, 4]])
    values = np.array([7, 8])
    shape = np.array([5, 6])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(9)
    name = "sparse_fill_7"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[1, 1]])
    values = np.array([-10])
    shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0)
    name = "sparse_fill_8"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(-5)
    name = "sparse_fill_9"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0]])
    values = np.array([1])
    shape = np.array([4, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(1)
    name = "sparse_fill_10"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    indices = np.array([[0, 0], [2, 2], [4, 4]])
    values = np.array([1, 2, 3], dtype=np.int64)
    shape = np.array([5, 5])
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=tf.int64)
    name = "sparse_fill_11"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float64)
    shape = np.array([5, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=tf.float64)
    name = "sparse_fill_12"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["tf.sparse.fill_empty_rows"] = tf_sparse_fill_empty_rows_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.fill_empty_rows' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.fill_empty_rows'.")

check_valid('tf.sparse.fill_empty_rows', generated_inputs['tf.sparse.fill_empty_rows'], lib="tf", suffix=0)
