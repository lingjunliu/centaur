
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_softmax_inputs():
    list_of_inputs = []

    def get_range(sp_tensor):
        return [np.min(sp_tensor.values), np.max(sp_tensor.values)] if sp_tensor.values.size > 0 else [0, 0]

    # Input 1: Basic example
    indices = np.array([[0, 0, 1], [0, 1, 0]])
    values = np.array([1., 1.], np.float32)
    dense_shape = np.array([1, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with larger values
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    values = np.array([2.0, 3.0, 1.0, 4.0], np.float32)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": "softmax_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with some empty rows/columns
    indices = np.array([[0, 0, 0], [1, 1, 1]])
    values = np.array([1.0, 2.0], np.float32)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different arrangement of indices
    indices = np.array([[0, 1, 0], [1, 0, 1]])
    values = np.array([1.0, 2.0], np.float32)
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different values
    indices = np.array([[0, 0, 1], [0, 1, 0]])
    values = np.array([5. , 2.], np.float32)
    dense_shape = np.array([1, 2, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: example with batch size > 1
    indices = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
    values = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    dense_shape = np.array([2, 1, 2])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    input_dict = {"sp_input": sp_input, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.softmax"] = tf_sparse_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.softmax'.")

check_valid('tf.sparse.softmax', generated_inputs['tf.sparse.softmax'], lib="tf", suffix=0)
