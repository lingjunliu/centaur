
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_maximum_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.sparse.SparseTensor(indices, values, dense_shape)

    def get_tensor_size(tensor):
        if isinstance(tensor, tf.sparse.SparseTensor):
            return tensor.values.size
        elif isinstance(tensor, np.ndarray):
            return tensor.size
        else:
            return tf.size(tensor).numpy()

    # Input 1
    indices_a = np.array([[0], [2]])
    values_a = np.array([1, 3], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2]])
    values_b = np.array([2, 4], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([5, 7], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 1]])
    values_b = np.array([6, 8], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_a = np.array([[0], [1], [2]])
    values_a = np.array([1, 2, 3], dtype=np.int32)
    dense_shape_a = np.array([4])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2], [3]])
    values_b = np.array([4, 5, 6], dtype=np.int32)
    dense_shape_b = np.array([4])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_overlap"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices_a = np.array([[0, 0, 0], [1, 1, 1]])
    values_a = np.array([10, 20], dtype=np.int32)
    dense_shape_a = np.array([2, 2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0, 1], [1, 1, 1]])
    values_b = np.array([11, 21], dtype=np.int32)
    dense_shape_b = np.array([2, 2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - negative values
    indices_a = np.array([[0], [2]])
    values_a = np.array([-1, 3], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2]])
    values_b = np.array([2, -4], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - all values present
    indices_a = np.array([[0], [1], [2], [3], [4]])
    values_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [1], [2], [3], [4]])
    values_b = np.array([6, 7, 8, 9, 10], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_all_present"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices_a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_a = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_b = np.array([5, 6, 7, 8], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices_a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - different shapes with same dense_shape
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]])
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_diff_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - empty sparse tensor
    indices_a = np.array([], dtype=np.int64).reshape(0, 2)
    values_a = np.array([], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = create_sparse_tensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = create_sparse_tensor(indices_b, values_b, dense_shape_b)
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "max_sparse_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.maximum"] = tf_sparse_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.maximum'.")

check_valid('tf.sparse.maximum', generated_inputs['tf.sparse.maximum'], lib="tf", suffix=0)
