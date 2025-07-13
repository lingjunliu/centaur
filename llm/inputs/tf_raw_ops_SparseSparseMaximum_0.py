
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSparseMaximum_inputs():
    list_of_inputs = []

    # Input 1
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, 2.0], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([3.0, 4.0], dtype=np.float32)
    b_shape = np.array([2, 3], dtype=np.int64)
    name = None
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.float32), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.float32), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 2
    a_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    a_values = np.array([5, 6], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([7, 8], dtype=np.int32)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_max_2"
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.int32), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.int32), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 3
    a_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    a_values = np.array([9.0, 10.0], dtype=np.float64)
    a_shape = np.array([2, 2, 2], dtype=np.int64)
    b_indices = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
    b_values = np.array([11.0, 12.0], dtype=np.float64)
    b_shape = np.array([2, 2, 2], dtype=np.int64)
    name = None
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.float64), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.float64), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 4
    a_indices = np.array([[0, 0]], dtype=np.int64)
    a_values = np.array([13], dtype=np.int64)
    a_shape = np.array([1, 1], dtype=np.int64)
    b_indices = np.array([[0, 0]], dtype=np.int64)
    b_values = np.array([14], dtype=np.int64)
    b_shape = np.array([1, 1], dtype=np.int64)
    name = "sparse_max_4"
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.int64), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.int64), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 5
    a_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    a_values = np.array([15, 16], dtype=np.uint8)
    a_shape = np.array([1, 2], dtype=np.int64)
    b_indices = np.array([[0, 1], [0, 0]], dtype=np.int64)
    b_values = np.array([17, 18], dtype=np.uint8)
    b_shape = np.array([1, 2], dtype=np.int64)
    name = None
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.uint8), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.uint8), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

   # Input 6
    a_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    a_values = np.array([19, 20], dtype=np.int16)
    a_shape = np.array([2, 1], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    b_values = np.array([21, 22], dtype=np.int16)
    b_shape = np.array([2, 1], dtype=np.int64)
    name = "sparse_max_6"
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.int16), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.int16), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 7
    a_indices = np.array([[0, 0, 0]], dtype=np.int64)
    a_values = np.array([23], dtype=np.int8)
    a_shape = np.array([1, 1, 1], dtype=np.int64)
    b_indices = np.array([[0, 0, 0]], dtype=np.int64)
    b_values = np.array([24], dtype=np.int8)
    b_shape = np.array([1, 1, 1], dtype=np.int64)
    name = None
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.int8), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.int8), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 8
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([25.0, 26.0], dtype=np.float16)
    a_shape = np.array([2, 2], dtype=np.int64)
    b_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    b_values = np.array([27.0, 28.0], dtype=np.float16)
    b_shape = np.array([2, 2], dtype=np.int64)
    name = "sparse_max_8"
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.float16), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.float16), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 9
    a_indices = np.array([[0, 0]], dtype=np.int64)
    a_values = np.array([29], dtype=np.uint16)
    a_shape = np.array([1, 1], dtype=np.int64)
    b_indices = np.array([[0, 0]], dtype=np.int64)
    b_values = np.array([30], dtype=np.uint16)
    b_shape = np.array([1, 1], dtype=np.int64)
    name = None
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.uint16), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.uint16), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    # Input 10
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([31.0, 32.0], dtype=np.half)
    a_shape = np.array([2, 3], dtype=np.int64)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    b_values = np.array([33.0, 34.0], dtype=np.half)
    b_shape = np.array([2, 3], dtype=np.int64)
    name = "sparse_max_10"
    input_dict = {"a_indices": tf.convert_to_tensor(a_indices, dtype=tf.int64), "a_values": tf.convert_to_tensor(a_values, dtype=tf.float16), "a_shape": tf.convert_to_tensor(a_shape, dtype=tf.int64), "b_indices": tf.convert_to_tensor(b_indices, dtype=tf.int64), "b_values": tf.convert_to_tensor(b_values, dtype=tf.float16), "b_shape": tf.convert_to_tensor(b_shape, dtype=tf.int64), "name": name}
    list_of_inputs.append({"args":[], "kwargs": input_dict})

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSparseMaximum"] = tf_raw_ops_SparseSparseMaximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSparseMaximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSparseMaximum'.")

check_valid('tf.raw_ops.SparseSparseMaximum', generated_inputs['tf.raw_ops.SparseSparseMaximum'], lib="tf", suffix=0)
