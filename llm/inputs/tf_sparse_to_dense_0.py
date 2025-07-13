
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_to_dense_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    dense_shape = np.array([3, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = True
    name = "sparse_to_dense_1"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different default value
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    dense_shape = np.array([3, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(-1, dtype=np.int64)
    validate_indices = False
    name = "sparse_to_dense_2"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dtype
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1.5, 2.5], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0.0, dtype=np.float32)
    validate_indices = True
    name = "sparse_to_dense_3"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    indices = np.array([[0, 0, 0], [1, 2, 1]])
    values = np.array([1, 2])
    dense_shape = np.array([2, 3, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = False
    name = "sparse_to_dense_4"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor
    indices = np.array([[0], [2]])
    values = np.array([1, 2])
    dense_shape = np.array([5])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = True
    name = "sparse_to_dense_5"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bool values
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([True, False])
    dense_shape = np.array([3, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(False, dtype=tf.bool)
    validate_indices = True
    name = "sparse_to_dense_7"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Large shape
    indices = np.array([[0, 0], [99, 99]])
    values = np.array([1, 2])
    dense_shape = np.array([100, 100])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = True
    name = "sparse_to_dense_8"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([-1, -2])
    dense_shape = np.array([3, 4])
    sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    default_value = tf.constant(0, dtype=np.int64)
    validate_indices = False
    name = "sparse_to_dense_9"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String type. Remove due to string type
    # indices = np.array([[0, 0], [1, 2]])
    # values = np.array(["a", "b"])
    # dense_shape = np.array([3, 4])
    # sp_input = tf.sparse.SparseTensor(indices, values, dense_shape)
    # default_value = tf.constant("", dtype=tf.string)
    # validate_indices = True
    # name = "sparse_to_dense_10"
    # input_dict = {"sp_input": sp_input, "default_value": default_value, "validate_indices": validate_indices, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.to_dense"] = tf_sparse_to_dense_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.to_dense' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_dense'.")

check_valid('tf.sparse.to_dense', generated_inputs['tf.sparse.to_dense'], lib="tf", suffix=0)
