
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSparseMinimum_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.int32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3, 4], dtype=np.int32)
    b_shape = np.array([2, 3], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_1",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 values
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1.5, 2.5], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3.5, 4.5], dtype=np.float32)
    b_shape = np.array([2, 3], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_2",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different sparse structure, int64 values
    a_indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
    a_values = np.array([5, 6], dtype=np.int64)
    a_shape = np.array([3, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    b_values = np.array([7, 8], dtype=np.int64)
    b_shape = np.array([3, 2], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_3",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty SparseTensor
    a_indices = np.array([], dtype=np.int64).reshape(0, 2)
    a_values = np.array([], dtype=np.float64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([1.0, 2.0], dtype=np.float64)
    b_shape = np.array([2, 2], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_4",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Overlapping indices, uint8
    a_indices = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([1, 2, 3], dtype=np.uint8)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([4, 5], dtype=np.uint8)
    b_shape = np.array([2, 2], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_5",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: half
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.float16)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3, 4], dtype=np.float16)
    b_shape = np.array([2, 3], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_6",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: larger shape
    a_indices = np.array([[0, 0], [1, 2], [3, 1], [4, 0]], dtype=np.int64)
    a_values = np.array([1, 2, 3, 4], dtype=np.int32)
    a_shape = np.array([5, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1], [3, 1], [4, 2]], dtype=np.int64)
    b_values = np.array([5, 6, 7, 8], dtype=np.int32)
    b_shape = np.array([5, 3], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_8",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dtypes and values.
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([1.0, 2.0], dtype=np.float64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([0.5, 1.5], dtype=np.float64)
    b_shape = np.array([2, 2], dtype=np.int64)

    input_dict = {
        "name": "sparse_minimum_10",
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSparseMinimum"] = tf_raw_ops_SparseSparseMinimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSparseMinimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSparseMinimum'.")

check_valid('tf.raw_ops.SparseSparseMinimum', generated_inputs['tf.raw_ops.SparseSparseMinimum'], lib="tf", suffix=0)
