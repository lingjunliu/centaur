
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseDenseCwiseMul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    sp_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float32)
    sp_shape = np.array([2, 3], dtype=np.int64)
    dense = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, int32
    sp_indices = np.array([[0, 1], [2, 0], [2, 1]], dtype=np.int64)
    sp_values = np.array([1, 2, 3], dtype=np.int32)
    sp_shape = np.array([3, 2], dtype=np.int64)
    dense = np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int32)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D sparse tensor, float64
    sp_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    sp_values = np.array([1.5, 2.5], dtype=np.float64)
    sp_shape = np.array([2, 2, 2], dtype=np.int64)
    dense = np.array([[[3.5, 4.5], [5.5, 6.5]], [[7.5, 8.5], [9.5, 10.5]]], dtype=np.float64)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8, different values
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([10, 20], dtype=np.uint8)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[30, 40], [50, 60]], dtype=np.uint8)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64, shape 1x1
    sp_indices = np.array([[0, 0]], dtype=np.int64)
    sp_values = np.array([100], dtype=np.int64)
    sp_shape = np.array([1, 1], dtype=np.int64)
    dense = np.array([[200]], dtype=np.int64)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float32, different indices
    sp_indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Half
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.float16)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, 6]], dtype=np.float16)
    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1.0, 2.0], dtype=np.float64)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.int8)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, 6]], dtype=np.int8)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16
    sp_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    sp_values = np.array([1, 2], dtype=np.int16)
    sp_shape = np.array([2, 2], dtype=np.int64)
    dense = np.array([[3, 4], [5, 6]], dtype=np.int16)

    input_dict = {
        "sp_indices": sp_indices,
        "sp_values": sp_values,
        "sp_shape": sp_shape,
        "dense": dense,
        "name": "test_mul_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseDenseCwiseMul"] = tf_raw_ops_SparseDenseCwiseMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseDenseCwiseMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseDenseCwiseMul'.")

check_valid('tf.raw_ops.SparseDenseCwiseMul', generated_inputs['tf.raw_ops.SparseDenseCwiseMul'], lib="tf", suffix=0)
