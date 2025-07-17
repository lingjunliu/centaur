
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic example with dense tensors
    inp_0 = np.array([['a'], ['b']], dtype=np.string_)
    inp_1 = np.array([['c'], ['d']], dtype=np.string_)
    inputs = [tf.constant(inp_0), tf.constant(inp_1)]
    name = "cross_op_1"
    separator = "_Y_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Sparse tensors
    indices_0 = np.array([[0, 0], [1, 0]])
    values_0 = np.array(['a', 'b'], dtype=np.string_)
    shape_0 = np.array([2, 1])
    sparse_tensor_0 = tf.SparseTensor(indices_0, values_0, shape_0)

    indices_1 = np.array([[0, 0], [1, 0]])
    values_1 = np.array(['c', 'd'], dtype=np.string_)
    shape_1 = np.array([2, 1])
    sparse_tensor_1 = tf.SparseTensor(indices_1, values_1, shape_1)

    inputs = [sparse_tensor_0, sparse_tensor_1]
    name = "cross_op_2"
    separator = "_Z_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed sparse and dense tensors
    indices_0 = np.array([[0, 0], [1, 0]])
    values_0 = np.array(['a', 'b'], dtype=np.string_)
    shape_0 = np.array([2, 1])
    sparse_tensor_0 = tf.SparseTensor(indices_0, values_0, shape_0)

    inp_1 = np.array([['c'], ['d']], dtype=np.string_)
    inputs = [sparse_tensor_0, tf.constant(inp_1)]
    name = "cross_op_3"
    separator = "_W_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three inputs - Reduce complexity for now
    inp_0 = np.array([['a'], ['b']], dtype=np.string_)
    inp_1 = np.array([['c'], ['d']], dtype=np.string_)

    inputs = [tf.constant(inp_0), tf.constant(inp_1)]
    name = "cross_op_4"
    separator = "_V_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes that can be broadcasted

    inp_0 = np.array([['a'], ['b']], dtype=np.string_)
    inp_1 = np.array([['c']], dtype=np.string_)
    inputs = [tf.constant(inp_0), tf.constant(inp_1)]
    name = "cross_op_6"
    separator = "_S_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: separator is None (empty string)

    inp_0 = np.array([['a'], ['b']], dtype=np.string_)
    inp_1 = np.array([['c'], ['d']], dtype=np.string_)
    inputs = [tf.constant(inp_0), tf.constant(inp_1)]
    name = "cross_op_7"
    separator = ""
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different sized dense and sparse tensors
    indices_0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int64)
    values_0 = np.array(['a', 'b', 'c'], dtype=np.string_)
    shape_0 = np.array([3, 1], dtype=np.int64)
    sparse_tensor_0 = tf.SparseTensor(indices_0, values_0, shape_0)
    inp_1 = np.array([['d'], ['e']], dtype=np.string_)
    inputs = [sparse_tensor_0, tf.constant(inp_1)]
    name = "cross_op_8"
    separator = "_Q_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 8: Dense tensors, different shapes
    inp_0 = np.array([['a', 'b'], ['c', 'd']], dtype=np.string_)
    inp_1 = np.array([['e'], ['f']], dtype=np.string_)
    inputs = [tf.constant(inp_0), tf.constant(inp_1)]
    name = "cross_op_9"
    separator = "_X_"
    input_dict = {"inputs": inputs, "name": name, "separator": separator}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross"] = tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
