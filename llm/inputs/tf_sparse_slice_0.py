
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_slice_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 2], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([2, 2])
    name = "slice1"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([5, 6, 7])
    shape = np.array([3, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([1, 0])
    size = np.array([2, 2])
    name = "slice2"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    values = np.array([8, 9, 10, 11])
    shape = np.array([4, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([4, 4])
    name = "slice3"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([12, 13, 14])
    shape = np.array([2, 5])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 2])
    size = np.array([2, 3])
    name = "slice4"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([15, 16])
    shape = np.array([5, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([2, 0])
    size = np.array([3, 3])
    name = "slice5"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0]])
    values = np.array([17])
    shape = np.array([1, 1])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([1, 1])
    name = "slice6"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([2, 2])
    name = "slice7"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0], [1, 0], [1, 1]])
    values = np.array([5, 6, 7])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([2, 1])
    name = "slice8"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([8, 9])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 0])
    size = np.array([1, 2])
    name = "slice9"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0], [1, 0], [1, 1]])
    values = np.array([10, 11, 12])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    start = np.array([0, 1])
    size = np.array([2, 1])
    name = "slice10"
    input_dict = {"sp_input": sp_input, "start": start, "size": size, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
inputs = tf_sparse_slice_inputs()
generated_inputs["tf.sparse.slice"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.slice'.")

check_valid('tf.sparse.slice', generated_inputs['tf.sparse.slice'], lib="tf", suffix=0)
