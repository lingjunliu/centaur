
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReshape_inputs():
    list_of_inputs = []

    # Input 1
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_shape = np.array([2, 5], dtype=np.int64)
    new_shape = np.array([1, 10], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_1", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    input_shape = np.array([1, 3], dtype=np.int64)
    new_shape = np.array([3, 1], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_2", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    input_shape = np.array([1, 2, 2], dtype=np.int64)
    new_shape = np.array([1, 4], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_3", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    input_shape = np.array([3, 5], dtype=np.int64)
    new_shape = np.array([5, 3], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_4", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_indices = np.array([[0, 0]], dtype=np.int64)
    input_shape = np.array([1, 1], dtype=np.int64)
    new_shape = np.array([1], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_5", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_indices = np.array([[0, 0, 0]], dtype=np.int64)
    input_shape = np.array([1, 1, 1], dtype=np.int64)
    new_shape = np.array([1], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_6", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([4], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_7", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    input_shape = np.array([3, 3, 3], dtype=np.int64)
    new_shape = np.array([27], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_8", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([1, 4], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_9", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_indices = np.array([[0, 0], [1, 0], [0, 1], [1,1]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([4], dtype=np.int64)
    input_dict = {"name": "sparse_reshape_10", "input_indices": input_indices, "input_shape": input_shape, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseReshape"] = tf_raw_ops_SparseReshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseReshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReshape'.")

check_valid('tf.raw_ops.SparseReshape', generated_inputs['tf.raw_ops.SparseReshape'], lib="tf", suffix=0)
