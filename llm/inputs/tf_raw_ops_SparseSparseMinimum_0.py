
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSparseMinimum_inputs():
    list_of_inputs = []

    # Input 1
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    b_values = np.array([3, 4], dtype=np.float32)
    b_shape = np.array([2, 3], dtype=np.int64)
    name = "sparse_minimum_1"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    a_values = np.array([5, 6], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    b_values = np.array([7, 8], dtype=np.int32)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_minimum_2"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    a_values = np.array([1.5, 2.5], dtype=np.float64)
    a_shape = np.array([2, 2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    b_values = np.array([3.5, 4.5], dtype=np.float64)
    b_shape = np.array([2, 2, 2], dtype=np.int64)
    name = "sparse_minimum_3"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([10, 20, 30, 40], dtype=np.int64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([15, 25, 35, 45], dtype=np.int64)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_minimum_4"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([-1, -2], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([1, -3], dtype=np.int32)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_minimum_5"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    a_indices = np.array([[0, 0]], dtype=np.int64)
    a_values = np.array([1], dtype=np.uint8)
    a_shape = np.array([1, 1], dtype=np.int64)
    b_indices = np.array([[0, 0]], dtype=np.int64)
    b_values = np.array([2], dtype=np.uint8)
    b_shape = np.array([1, 1], dtype=np.int64)
    name = "sparse_minimum_6"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a_indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.float16)
    a_shape = np.array([1, 2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    b_values = np.array([3, 4], dtype=np.float16)
    b_shape = np.array([1, 2, 2], dtype=np.int64)
    name = "sparse_minimum_7"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([1j, 2j], dtype=np.complex64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3j, 4j], dtype=np.complex64)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_minimum_8"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a_indices = np.array([[0, 0]], dtype=np.int64)
    a_values = np.array([1], dtype=np.uint32)
    a_shape = np.array([1, 1], dtype=np.int64)
    b_indices = np.array([[0, 0]], dtype=np.int64)
    b_values = np.array([2], dtype=np.uint32)
    b_shape = np.array([1, 1], dtype=np.int64)
    name = "sparse_minimum_9"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3, 4], dtype=np.float32)
    b_shape = np.array([2, 3], dtype=np.int64)
    name = "sparse_minimum_10"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b_indices": b_indices,
        "b_values": b_values,
        "b_shape": b_shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_SparseSparseMinimum_inputs()
generated_inputs["tf.raw_ops.SparseSparseMinimum"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.SparseSparseMinimum"].append({k: tf.constant(v) if k != "name" else v for k, v in input_dict.items()})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSparseMinimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSparseMinimum'.")

check_valid('tf.raw_ops.SparseSparseMinimum', generated_inputs['tf.raw_ops.SparseSparseMinimum'], lib="tf", suffix=0)
