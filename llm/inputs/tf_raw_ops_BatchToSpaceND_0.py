
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_batch_to_space_nd_inputs():
    list_of_inputs = []

    # Input 1: Simple example from documentation
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another simple example from documentation
    input_tensor = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with crops
    input_tensor = np.array([[[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0], [10], [12]]], [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6], [8]]], [[[0], [14], [16]]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [2, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with block_shape > 1 and crops = 0
    input_tensor = np.random.randint(0, 10, size=(8, 2, 2, 1), dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type
    input_tensor = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]], dtype=np.float32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6:  Larger block_shape
    input_tensor = np.random.randint(0, 10, size=(16, 1, 1, 1), dtype=np.int32)
    block_shape = np.array([4, 4], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different crops
    input_tensor = np.random.randint(0, 10, size=(4, 2, 2, 1), dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D spatial shape, different block shapes
    input_tensor = np.random.randint(0, 10, size=(8, 1, 2, 3, 1), dtype=np.int32)
    block_shape = np.array([2, 1, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 type
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int64)
    block_shape = np.array([2, 2], dtype=np.int64)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int64)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger crops, still valid
    input_tensor = np.random.randint(0, 10, size=(4, 2, 2, 1), dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[1, 1], [1, 1]], dtype=np.int32)
    input_dict = {"input": input_tensor, "block_shape": block_shape, "crops": crops, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchToSpaceND"] = tf_raw_ops_batch_to_space_nd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchToSpaceND' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchToSpaceND'.")

check_valid('tf.raw_ops.BatchToSpaceND', generated_inputs['tf.raw_ops.BatchToSpaceND'], lib="tf", suffix=0)
