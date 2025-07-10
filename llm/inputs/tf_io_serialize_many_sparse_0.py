
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic valid SparseTensor
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_1"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SparseTensor with different shape and values
    indices = np.array([[0, 0], [0, 2], [1, 1]])
    values = np.array([5, 6, 7])
    shape = np.array([2, 3])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_2"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: SparseTensor with larger shape
    indices = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
    values = np.array([8, 9, 10, 11])
    shape = np.array([4, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_3"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: SparseTensor with rank 3
    indices = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_5"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: SparseTensor with float values
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1.0, 2.0, 3.0])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_7"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 2, batch size 1
    indices = np.array([[0, 0], [0, 1]])
    values = np.array([1, 2])
    shape = np.array([1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_10"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: With negative values
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([-1, 2, -3])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_13"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: empty
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_14"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bool values
    indices = np.array([[0, 0], [0, 1]])
    values = np.array([True, False])
    shape = np.array([1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_15"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 5
    indices = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 1, 1, 1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_16"
    input_dict = {"sp_input": sp_input, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_many_sparse"] = tf_io_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
