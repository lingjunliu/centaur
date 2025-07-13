
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_split_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    def get_tensor_size(tensor):
        if isinstance(tensor, tf.sparse.SparseTensor):
            return tensor.values.size
        elif isinstance(tensor, np.ndarray):
            return tensor.size
        else:
            return tensor.numpy().size if hasattr(tensor, 'numpy') else 1

    # Input 1
    indices = np.array([[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dense_shape = np.array([2, 7], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int32)
    name = "split_1"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [1, 2], [1, 3]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    dense_shape = np.array([2, 4], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 3
    axis = np.array(1, dtype=np.int32)
    name = "split_2"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 2
    axis = np.array(0, dtype=np.int32)
    name = "split_3"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 2
    axis = np.array(0, dtype=np.int32)
    name = "split_4"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 2
    axis = np.array(1, dtype=np.int32)
    name = "split_5"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 2
    axis = np.array(-1, dtype=np.int32)
    name = "split_6"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 1
    axis = np.array(0, dtype=np.int32)
    name = "split_7"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dense_shape = np.array([3, 3, 3], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 3
    axis = np.array(2, dtype=np.int32)
    name = "split_8"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 3
    axis = np.array(1, dtype=np.int32)
    name = "split_9"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = np.array([3, 2], dtype=np.int64)
    sp_input = create_sparse_tensor(indices, values, dense_shape)
    num_split = 3
    axis = np.array(0, dtype=np.int32)
    name = "split_10"

    input_dict = {
        "sp_input": sp_input,
        "num_split": num_split,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.split"] = tf_sparse_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.split'.")

check_valid('tf.sparse.split', generated_inputs['tf.sparse.split'], lib="tf", suffix=0)
