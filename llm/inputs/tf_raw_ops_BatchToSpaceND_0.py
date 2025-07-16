
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_batch_to_space_nd_inputs():
    list_of_inputs = []

    # Input 1: Simple case
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"name": "input1", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type, float32
    input_tensor = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]], dtype=np.float32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"name": "input2", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Crops applied
    input_tensor = np.array([[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0], [10], [12]]],
                           [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6], [8]]], [[[0], [14], [16]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [2, 0]], dtype=np.int32)
    input_dict = {"name": "input3", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shape
    input_tensor = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"name": "input4", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Block shape as int64
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int64)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"name": "input5", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: crops as int64
    input_tensor = np.array([[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0], [10], [12]]],
                           [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6], [8]]], [[[0], [14], [16]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [2, 0]], dtype=np.int64)
    input_dict = {"name": "input6", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: 3D input
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    block_shape = np.array([1], dtype=np.int32)
    crops = np.array([[0, 0]], dtype=np.int32)
    input_dict = {"name": "input7", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D input
    input_tensor = np.random.randint(0, 10, size=(4, 1, 1, 1, 1), dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"name": "input8", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger input
    input_tensor = np.random.randint(0, 10, size=(16, 2, 2, 3), dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {"name": "input9", "input": input_tensor, "block_shape": block_shape, "crops": crops}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different crop values
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {"name": "input10", "input": input_tensor, "block_shape": block_shape, "crops": crops}
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
