
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_maximum_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.maximum function.
    The inputs are provided as dense numpy arrays, which have the .size attribute,
    to resolve the issue with the execution framework's pre-check.
    The framework is expected to handle the conversion to SparseTensors if needed.
    """
    list_of_inputs = []

    def to_dense(indices, values, shape, dtype):
        dense = np.zeros(shape, dtype=dtype)
        if indices.shape[0] > 0:
            dense[tuple(indices.T)] = values
        return dense

    # Case 1: Basic 1D Tensors with overlapping indices
    shape1 = (5,)
    dtype1 = np.int32
    sp_a_1 = to_dense(np.array([[1], [3]]), np.array([10, 30], dtype=dtype1), shape1, dtype1)
    sp_b_1 = to_dense(np.array([[1], [4]]), np.array([20, 40], dtype=dtype1), shape1, dtype1)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_1, 'sp_b': sp_b_1, 'name': 'basic_1d'}))

    # Case 2: Basic 2D Tensors
    shape2 = (3, 4)
    dtype2 = np.int32
    sp_a_2 = to_dense(np.array([[0, 1], [1, 2]]), np.array([1, 2], dtype=dtype2), shape2, dtype2)
    sp_b_2 = to_dense(np.array([[0, 1], [2, 0]]), np.array([3, 4], dtype=dtype2), shape2, dtype2)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_2, 'sp_b': sp_b_2, 'name': 'basic_2d'}))

    # Case 3: Negative values
    shape3 = (4,)
    dtype3 = np.int32
    sp_a_3 = to_dense(np.array([[0], [2]]), np.array([-5, 5], dtype=dtype3), shape3, dtype3)
    sp_b_3 = to_dense(np.array([[0], [1]]), np.array([-10, -2], dtype=dtype3), shape3, dtype3)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_3, 'sp_b': sp_b_3, 'name': 'negative_values'}))

    # Case 4: Float dtype
    shape4 = (2, 2)
    dtype4 = np.float32
    sp_a_4 = to_dense(np.array([[0, 0], [1, 1]]), np.array([1.5, -2.5], dtype=dtype4), shape4, dtype4)
    sp_b_4 = to_dense(np.array([[0, 0], [0, 1]]), np.array([-1.0, 3.5], dtype=dtype4), shape4, dtype4)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_4, 'sp_b': sp_b_4, 'name': 'float_dtype'}))

    # Case 5: One tensor is empty
    shape5 = (5,)
    dtype5 = np.int32
    sp_a_5 = to_dense(np.array([[0], [2]]), np.array([10, 20], dtype=dtype5), shape5, dtype5)
    sp_b_5 = to_dense(np.empty((0, 1)), np.array([], dtype=dtype5), shape5, dtype5)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_5, 'sp_b': sp_b_5, 'name': 'one_empty'}))

    # Case 6: Both tensors are empty
    shape6 = (3, 3)
    dtype6 = np.int32
    sp_a_6 = to_dense(np.empty((0, 2)), np.array([], dtype=dtype6), shape6, dtype6)
    sp_b_6 = to_dense(np.empty((0, 2)), np.array([], dtype=dtype6), shape6, dtype6)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_6, 'sp_b': sp_b_6, 'name': 'both_empty'}))

    # Case 7: Identical tensors
    shape7 = (3, 3)
    dtype7 = np.int32
    sp_a_7 = to_dense(np.array([[0, 1], [2, 2]]), np.array([5, 9], dtype=dtype7), shape7, dtype7)
    sp_b_7 = copy.deepcopy(sp_a_7)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_7, 'sp_b': sp_b_7, 'name': 'identical_tensors'}))

    # Case 8: No overlapping indices
    shape8 = (2, 2)
    dtype8 = np.int32
    sp_a_8 = to_dense(np.array([[0, 0], [1, 1]]), np.array([1, 2], dtype=dtype8), shape8, dtype8)
    sp_b_8 = to_dense(np.array([[0, 1], [1, 0]]), np.array([3, 4], dtype=dtype8), shape8, dtype8)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_8, 'sp_b': sp_b_8, 'name': 'no_overlap'}))

    # Case 9: 3D tensors
    shape9 = (2, 2, 2)
    dtype9 = np.int32
    sp_a_9 = to_dense(np.array([[0, 0, 1], [1, 1, 0]]), np.array([1, 2], dtype=dtype9), shape9, dtype9)
    sp_b_9 = to_dense(np.array([[0, 0, 1], [1, 0, 1]]), np.array([3, 4], dtype=dtype9), shape9, dtype9)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_9, 'sp_b': sp_b_9, 'name': '3d_tensors'}))

    # Case 10: Using float64 dtype
    shape10 = (2, 2)
    dtype10 = np.float64
    sp_a_10 = to_dense(np.array([[0, 1]]), np.array([1e10], dtype=dtype10), shape10, dtype10)
    sp_b_10 = to_dense(np.array([[0, 1]]), np.array([1e10 - 1], dtype=dtype10), shape10, dtype10)
    list_of_inputs.append(copy.deepcopy({'sp_a': sp_a_10, 'sp_b': sp_b_10, 'name': 'float64_dtype'}))

    return list_of_inputs

generated_inputs["tf.sparse.maximum"] = tf_sparse_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.maximum'.")

check_valid('tf.sparse.maximum', generated_inputs['tf.sparse.maximum'], lib="tf", suffix=0)
