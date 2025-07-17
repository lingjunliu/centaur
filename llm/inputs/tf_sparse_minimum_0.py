
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_minimum_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.SparseTensor(indices, values, dense_shape)

    def get_tensor_size(tensor):
        if isinstance(tensor, tf.SparseTensor):
            return np.prod(tensor.dense_shape.numpy()) if tensor.dense_shape.numpy().size > 0 else 0
        else:
            return tensor.size

    # Input 1
    indices_a = np.array([[0, 0], [1, 2]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([3, 4])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 2]])
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([3, 4])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_a = np.array([[0], [2]])
    values_a = np.array([5, 6], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2]])
    values_b = np.array([7, 8], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_a = np.array([[0, 0, 0], [1, 1, 1]])
    values_a = np.array([9, 10], dtype=np.int32)
    dense_shape_a = np.array([2, 2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0, 0], [1, 1, 1]])
    values_b = np.array([11, 12], dtype=np.int32)
    dense_shape_b = np.array([2, 2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices_a = np.array([[0, 1], [1, 0]])
    values_a = np.array([-1, 2], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]])
    values_b = np.array([3, -4], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices_a = np.array([[0, 0]])
    values_a = np.array([1], dtype=np.int32)
    dense_shape_a = np.array([1, 1])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0]])
    values_b = np.array([1], dtype=np.int32)
    dense_shape_b = np.array([1, 1])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Empty SparseTensor
    indices_a = np.array([], dtype=np.int64).reshape(0, 2)
    values_a = np.array([], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([], dtype=np.int64).reshape(0, 2)
    values_b = np.array([], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Different values, same indices
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([10, 20], dtype=np.int32)
    dense_shape_a = np.array([3, 3])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 1]])
    values_b = np.array([5, 15], dtype=np.int32)
    dense_shape_b = np.array([3, 3])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.minimum"] = tf_sparse_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.minimum'.")

check_valid('tf.sparse.minimum', generated_inputs['tf.sparse.minimum'], lib="tf", suffix=0)
