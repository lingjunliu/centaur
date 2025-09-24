
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_batch_to_space_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example1"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3]], [[4, 5, 6]]], [[[7, 8, 9]], [[10, 11, 12]]]], dtype=np.int32)
    block_shape = np.array([2, 1], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example2"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]], [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]] , dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example3"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0], [10], [12]]], [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6], [8]]], [[[0], [14], [16]]]] , dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [2, 0]], dtype=np.int32)
    name = "example4"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (int64)
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int64)
    block_shape = np.array([2, 2], dtype=np.int64)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int64)
    name = "example5"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int64),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int64),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int64),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Different crops
    input_tensor = np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], dtype=np.int32)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[1, 0], [0, 1]], dtype=np.int32)
    name = "example6"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D input, 1D block_shape, 2D Crops
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    block_shape = np.array([2], dtype=np.int32)
    crops = np.array([[0, 0]], dtype=np.int32)
    name = "example7"
    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: block shape is 1
    input_tensor = np.array([[[[1]]], [[[2]]]], dtype=np.int32)
    block_shape = np.array([1, 1], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example8"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger input
    input_tensor = np.arange(1, 65, dtype=np.int32).reshape(8, 1, 2, 4)
    block_shape = np.array([2, 2], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example9"
    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D tensor
    input_tensor = np.arange(1, 33, dtype=np.int32).reshape(4, 1, 2, 1, 4)
    block_shape = np.array([2, 1], dtype=np.int32)
    crops = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "example10"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "block_shape": tf.convert_to_tensor(block_shape, dtype=tf.int32),
        "crops": tf.convert_to_tensor(crops, dtype=tf.int32),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.batch_to_space"] = tf_batch_to_space_inputs()

for i in range(len(generated_inputs["tf.batch_to_space"])):
    generated_inputs["tf.batch_to_space"][i]["input"] = generated_inputs["tf.batch_to_space"][i]["input"].numpy()
    generated_inputs["tf.batch_to_space"][i]["block_shape"] = generated_inputs["tf.batch_to_space"][i]["block_shape"].numpy()
    generated_inputs["tf.batch_to_space"][i]["crops"] = generated_inputs["tf.batch_to_space"][i]["crops"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.batch_to_space' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.batch_to_space'.")

check_valid('tf.batch_to_space', generated_inputs['tf.batch_to_space'], lib="tf", suffix=0)
