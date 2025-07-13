
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseTensorDenseAdd_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, 2.0], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with int32
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    a_values = np.array([1, 2], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int32)
    b = np.array([[3, 4], [5, 6]], dtype=np.int32)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shape
    a_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    a_values = np.array([1.0, 2.0], dtype=np.float64)
    a_shape = np.array([2, 2, 2], dtype=np.int64)
    b = np.array([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]], dtype=np.float64)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values
    a_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    a_values = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    a_shape = np.array([3, 3], dtype=np.int64)
    b = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]], dtype=np.float32)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element sparse tensor
    a_indices = np.array([[0, 0]], dtype=np.int32)
    a_values = np.array([5], dtype=np.int32)
    a_shape = np.array([1, 1], dtype=np.int32)
    b = np.array([[2]], dtype=np.int32)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using int64 values
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    a_values = np.array([1000000000, 2000000000], dtype=np.int64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[3, 4], [5, 6]], dtype=np.int64)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: uint8
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    a_values = np.array([1, 2], dtype=np.uint8)
    a_shape = np.array([2, 2], dtype=np.int32)
    b = np.array([[3, 4], [5, 6]], dtype=np.uint8)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: bfloat16. Reduce num of dimensions for this case
    a_indices = np.array([[0], [1]], dtype=np.int32)
    a_values = np.array([1, 2], dtype=np.float32).astype(np.float16)
    a_shape = np.array([2], dtype=np.int32)
    b = np.array([3, 4], dtype=np.float32).astype(np.float16)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: complex64
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    a_values = np.array([1+1j, 2+2j], dtype=np.complex64)
    a_shape = np.array([2, 2], dtype=np.int32)
    b = np.array([[3+3j, 4+4j], [5+5j, 6+6j]], dtype=np.complex64)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty sparse tensor
    a_indices = np.array([], dtype=np.int32).reshape(0, 2)
    a_values = np.array([], dtype=np.float32)
    a_shape = np.array([2, 2], dtype=np.int32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a_indices": tf.constant(a_indices), "a_values": tf.constant(a_values), "a_shape": tf.constant(a_shape), "b": tf.constant(b), "name": "add_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseTensorDenseAdd"] = tf_raw_ops_SparseTensorDenseAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseTensorDenseAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseTensorDenseAdd'.")

check_valid('tf.raw_ops.SparseTensorDenseAdd', generated_inputs['tf.raw_ops.SparseTensorDenseAdd'], lib="tf", suffix=0)
