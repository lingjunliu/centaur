
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_batch_to_space_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_1"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=np.int32)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_2"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]], dtype=np.float64)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int64)
    block_size_val = 2
    name_val = "batch_to_space_3"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
                            [[[10], [11], [12]], [[13], [14], [15]], [[16], [17], [18]]],
                            [[[19], [20], [21]], [[22], [23], [24]], [[25], [26], [27]]],
                            [[[28], [29], [30]], [[31], [32], [33]], [[34], [35], [36]]]], dtype=np.int32)
    crops_tensor = np.array([[1, 1], [1, 1]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_4"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
                            [[[25, 26, 27], [28, 29, 30]], [[31, 32, 33], [34, 35, 36]]],
                            [[[37, 38, 39], [40, 41, 42]], [[43, 44, 45], [46, 47, 48]]]], dtype=np.float32)
    crops_tensor = np.array([[0, 1], [1, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_5"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]], [[[7]], [[8]], [[9]]], [[[10]], [[11]], [[12]]]], dtype=np.int64)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int64)
    block_size_val = 2
    name_val = "batch_to_space_6"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=np.int32)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_7"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.arange(16).reshape((4, 1, 1, 4)).astype(np.int32)
    crops_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_9"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=np.int32)
    crops_tensor = np.array([[1, 0], [0, 1]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_10"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int32)
    crops_tensor = np.array([[0, 1], [1, 0]], dtype=np.int32)
    block_size_val = 2
    name_val = "batch_to_space_11"
    input_dict = {"input": input_tensor, "crops": crops_tensor, "block_size": block_size_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_batch_to_space_inputs()
generated_inputs["tf.raw_ops.BatchToSpace"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchToSpace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchToSpace'.")

check_valid('tf.raw_ops.BatchToSpace', generated_inputs['tf.raw_ops.BatchToSpace'], lib="tf", suffix=0)
