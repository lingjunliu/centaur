
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sequence_mask_inputs():
    list_of_inputs = []

    # Input 1
    lengths = np.array([1, 3, 2])
    maxlen = 5
    dtype = tf.bool
    name = "mask1"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    lengths = np.array([[1, 3], [2, 0]])
    maxlen = 4
    dtype = tf.bool
    name = "mask2"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    lengths = np.array([5])
    maxlen = 5
    dtype = tf.bool
    name = "mask3"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    lengths = np.array([0, 0, 0])
    maxlen = 3
    dtype = tf.bool
    name = "mask4"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    lengths = np.array([1, 2, 3, 4, 5])
    maxlen = 5
    dtype = tf.bool
    name = "mask5"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    lengths = np.array([[4, 2, 1], [3, 5, 0]])
    maxlen = 6
    dtype = tf.bool
    name = "mask6"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    lengths = np.array([2, 4, 1])
    maxlen = 4
    dtype = tf.float32
    name = "mask7"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    lengths = np.array([[1, 2], [3, 1]])
    maxlen = 3
    dtype = tf.int32
    name = "mask8"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    lengths = np.array([1, 2, 3])
    maxlen = 3
    dtype = tf.bool
    name = "mask9"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    lengths = np.array([1, 3, 2], dtype=np.int64)
    maxlen = 5
    dtype = tf.bool
    name = "mask10"
    input_dict = {"lengths": lengths, "maxlen": maxlen, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sequence_mask"] = tf_sequence_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sequence_mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sequence_mask'.")

check_valid('tf.sequence_mask', generated_inputs['tf.sequence_mask'], lib="tf", suffix=0)
