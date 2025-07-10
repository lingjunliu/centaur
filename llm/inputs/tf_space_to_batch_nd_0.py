
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_space_to_batch_nd_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]]).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0]]).astype(np.int32)
    name = "example_1"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0]]).astype(np.int32)
    name = "example_2"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0]]).astype(np.int32)
    name = "example_3"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1], [2], [3], [4]], [[5], [6], [7], [8]]], [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [2, 0]]).astype(np.int32)
    name = "example_4"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More dimensions
    input_tensor = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    block_shape = np.array([2, 2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0], [0, 0]]).astype(np.int32)
    name = "example_6"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger paddings
    input_tensor = np.random.rand(1, 2, 2, 1).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[2, 2], [2, 2]]).astype(np.int32)
    name = "example_7"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]]).astype(np.int32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0]]).astype(np.int32)
    name = "example_8"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another dimension
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    block_shape = np.array([4, 4]).astype(np.int32)
    paddings = np.array([[0, 0], [0, 0]]).astype(np.int32)
    name = "example_9"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: non-zero padding
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    block_shape = np.array([2, 2]).astype(np.int32)
    paddings = np.array([[1, 1], [1, 1]]).astype(np.int32)
    name = "example_10"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: block_shape of one and padding
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    block_shape = np.array([1, 1]).astype(np.int32)
    paddings = np.array([[1, 1], [1, 1]]).astype(np.int32)
    name = "example_12"
    input_dict = {"input": input_tensor, "block_shape": block_shape, "paddings": paddings, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.space_to_batch_nd"] = tf_space_to_batch_nd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.space_to_batch_nd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.space_to_batch_nd'.")

check_valid('tf.space_to_batch_nd', generated_inputs['tf.space_to_batch_nd'], lib="tf", suffix=0)
