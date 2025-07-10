
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic SparseTensor
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    dense_shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SparseTensor with different out_type
    indices = np.array([[0, 1], [2, 0]])
    values = np.array([3.14, 2.71])
    dense_shape = np.array([3, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty SparseTensor
    indices = np.array([[]], dtype=np.int64).reshape(0, 2)
    values = np.array([])
    dense_shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D SparseTensor
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([3, 3, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: SparseTensor with negative values
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([-1, -2])
    dense_shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: SparseTensor with large dense shape
    indices = np.array([[0, 0], [99, 99]])
    values = np.array([1, 2])
    dense_shape = np.array([100, 100])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: SparseTensor with float values and int indices/shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1.5, 2.5])
    dense_shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another Empty SparseTensor with different shape
    indices = np.array([[]], dtype=np.int64).reshape(0, 2)
    values = np.array([])
    dense_shape = np.array([5, 7])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D SparseTensor
    indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    values = np.array([1, 2])
    dense_shape = np.array([2, 2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: SparseTensor with zero value
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([0, 0])
    dense_shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_sparse"] = tf_io_serialize_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_sparse'.")

check_valid('tf.io.serialize_sparse', generated_inputs['tf.io.serialize_sparse'], lib="tf", suffix=0)
