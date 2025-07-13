
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReshape_inputs():
    list_of_inputs = []

    # Input 1: Basic reshape
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_shape = np.array([2, 3], dtype=np.int64)
    new_shape = np.array([3, 2], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Reshape with -1
    input_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    input_shape = np.array([1, 3], dtype=np.int64)
    new_shape = np.array([-1], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reshape from 3D to 2D
    input_indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    input_shape = np.array([1, 2, 2], dtype=np.int64)
    new_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Reshape from 2D to 3D
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([1, 2, 2], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Reshape with larger values in indices
    input_indices = np.array([[0, 0], [9, 9]], dtype=np.int64)
    input_shape = np.array([10, 10], dtype=np.int64)
    new_shape = np.array([100], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Reshape to a single element
    input_indices = np.array([[0, 0]], dtype=np.int64)
    input_shape = np.array([1, 1], dtype=np.int64)
    new_shape = np.array([1], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Reshape to multiple dimensions with -1
    input_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    input_shape = np.array([1, 1, 2], dtype=np.int64)
    new_shape = np.array([1, -1], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Reshape with a different arrangement
    input_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([4], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Reshape with different sizes
    input_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], dtype=np.int64)
    input_shape = np.array([3, 2], dtype=np.int64)
    new_shape = np.array([2, 3], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Reshape and keep the same shape
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    new_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Reshape with empty indices
    input_indices = np.array([], dtype=np.int64).reshape(0, 2)
    input_shape = np.array([2, 3], dtype=np.int64)
    new_shape = np.array([3, 2], dtype=np.int64)
    input_dict = {"input_indices": tf.constant(input_indices, dtype=tf.int64), "input_shape": tf.constant(input_shape, dtype=tf.int64), "new_shape": tf.constant(new_shape, dtype=tf.int64), "name": None}
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
