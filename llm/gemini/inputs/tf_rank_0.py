
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_rank_inputs():
    list_of_inputs = []

    # Input 1: 0-D tensor
    input_tensor = tf.constant(10).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D tensor
    input_tensor = tf.constant([1, 2, 3, 4, 5]).numpy()
    input_dict = {"input": input_tensor, "name": "rank_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D tensor
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3-D tensor
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {"input": input_tensor, "name": "rank_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4-D tensor
    input_tensor = tf.constant([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with different data type (float32)
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    input_dict = {"input": input_tensor, "name": "rank_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with different data type (int64)
    input_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int64).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Tensor with different shape
    input_tensor = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    input_tensor = tf.constant([]).numpy()
    input_dict = {"input": input_tensor, "name": "rank_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5-D tensor
    input_tensor = tf.constant([[[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]],[[[[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]]]]]).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.rank"] = tf_rank_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.rank' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.rank'.")

check_valid('tf.rank', generated_inputs['tf.rank'], lib="tf", suffix=0)
