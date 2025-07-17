
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_expand_dims_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.SparseTensor(indices=np.array(indices), values=np.array(values), dense_shape=np.array(dense_shape))

    # Input 1
    sp_input = create_sparse_tensor([[0, 0], [1, 2]], [1, 2], [3, 4])
    axis = 0
    name = "sparse_expanded"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sp_input = create_sparse_tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], [1, 2, 3], [3, 3, 3])
    axis = 1
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sp_input = create_sparse_tensor([[0], [1], [2]], [1, 2, 3], [3])
    axis = -1
    name = "sparse_expanded_neg"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sp_input = create_sparse_tensor([[0, 1], [2, 3]], [4, 5], [4, 4])
    axis = -2
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    sp_input = create_sparse_tensor([[0,0,0,0]], [7], [2,2,2,2])
    axis = 2
    name = "sparse_expanded_four_dim"
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sp_input = create_sparse_tensor([[0]], [8], [10])
    axis = 0
    name = None
    input_dict = {"sp_input": sp_input, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.expand_dims"] = tf_sparse_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.expand_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.expand_dims'.")

check_valid('tf.sparse.expand_dims', generated_inputs['tf.sparse.expand_dims'], lib="tf", suffix=0)
