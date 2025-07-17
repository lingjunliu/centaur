
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D SparseTensor, default permutation
    indices = np.array([[0, 1], [0, 3], [2, 3], [3, 1]])
    values = np.array([1.1, 2.2, 3.3, 4.4])
    dense_shape = np.array([4, 5])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_1"
    input_dict = {"sp_input": values, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D SparseTensor, custom permutation
    indices = np.array([[0, 1], [0, 3], [2, 3], [3, 1]])
    values = np.array([1.1, 2.2, 3.3, 4.4])
    dense_shape = np.array([4, 5])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [1, 0]
    name = "transpose_2"
    input_dict = {"sp_input": values, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D SparseTensor, default permutation
    indices = np.array([[0, 0, 1], [0, 0, 3], [1, 2, 3], [1, 3, 1]])
    values = np.array([1.1, 2.2, 3.3, 4.4])
    dense_shape = np.array([2, 4, 5])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_3"
    input_dict = {"sp_input": values, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.transpose"] = tf_sparse_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.transpose'.")

check_valid('tf.sparse.transpose', generated_inputs['tf.sparse.transpose'], lib="tf", suffix=0)
