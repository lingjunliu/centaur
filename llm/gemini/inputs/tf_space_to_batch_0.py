
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_space_to_batch_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]])
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_1"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_2"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - Modified to ensure divisibility and shape consistency. Corrected shape to (1,4,4,1) and adjusted values
    input_tensor = np.array([[[[1], [2], [3], [4]],[[5], [6], [7], [8]],[[9],[10],[11],[12]],[[13],[14],[15],[16]]]])
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_3"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1], [2], [3], [4]], [[5], [6], [7], [8]]], [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]])
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [2, 0]])
    name_str = "test_4"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 8, 8, 3)
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_5"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(2, 4, 4, 1)
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_6"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 16, 16, 1)
    block_shape_tensor = np.array([4, 4])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_7"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different block shape and paddings
    input_tensor = np.random.rand(1, 10, 10, 1)
    block_shape_tensor = np.array([5, 5])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_8"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different paddings with non-zero values
    input_tensor = np.random.rand(1, 5, 5, 1)
    block_shape_tensor = np.array([1, 1])
    paddings_tensor = np.array([[1, 1], [2, 2]])
    name_str = "test_9"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger batch size
    input_tensor = np.random.rand(4, 4, 4, 1)
    block_shape_tensor = np.array([2, 2])
    paddings_tensor = np.array([[0, 0], [0, 0]])
    name_str = "test_10"
    input_dict = {"input": input_tensor, "block_shape": block_shape_tensor, "paddings": paddings_tensor, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.space_to_batch"] = tf_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.space_to_batch'.")

check_valid('tf.space_to_batch', generated_inputs['tf.space_to_batch'], lib="tf", suffix=0)
