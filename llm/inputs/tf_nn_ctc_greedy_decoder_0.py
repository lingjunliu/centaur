
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_greedy_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2], [0.8, 0.1, 0.05, 0.05]],
                       [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    merge_repeated = True
    blank_index = 3
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
                       [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    merge_repeated = False
    blank_index = 0
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2], [0.8, 0.1, 0.05, 0.05]],
                       [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]],
                       [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]], dtype=np.float32)
    sequence_length = np.array([3, 3], dtype=np.int32)
    merge_repeated = True
    blank_index = 3
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2], [0.8, 0.1, 0.05, 0.05]],
                       [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]],
                       [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]], dtype=np.float32)
    sequence_length = np.array([3, 2], dtype=np.int32)
    merge_repeated = False
    blank_index = 0
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2]],
                       [[0.1, 0.1, 0.7, 0.1]]], dtype=np.float32)
    sequence_length = np.array([2], dtype=np.int32)
    merge_repeated = True
    blank_index = 3
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([[[0.7, 0.1, 0.1, 0.1]],
                       [[0.1, 0.1, 0.1, 0.7]]], dtype=np.float32)
    sequence_length = np.array([2], dtype=np.int32)
    merge_repeated = False
    blank_index = 0
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2, 0.0], [0.8, 0.1, 0.05, 0.05, 0.0]],
                       [[0.1, 0.1, 0.7, 0.1, 0.0], [0.1, 0.1, 0.1, 0.7, 0.0]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    merge_repeated = True
    blank_index = 4
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([[[0.7, 0.1, 0.1, 0.1, 0.0], [0.1, 0.7, 0.1, 0.1, 0.0]],
                       [[0.1, 0.1, 0.7, 0.1, 0.0], [0.1, 0.1, 0.1, 0.7, 0.0]]], dtype=np.float32)
    sequence_length = np.array([2, 2], dtype=np.int32)
    merge_repeated = False
    blank_index = -1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([[[0.1, 0.6, 0.1], [0.8, 0.1, 0.05]],
                       [[0.1, 0.1, 0.7], [0.1, 0.1, 0.1]],
                       [[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]]], dtype=np.float32)
    sequence_length = np.array([3, 2], dtype=np.int32)
    merge_repeated = True
    blank_index = 2
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([[[0.1, 0.6, 0.1], [0.8, 0.1, 0.05]],
                       [[0.1, 0.1, 0.7], [0.1, 0.1, 0.1]],
                       [[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]]], dtype=np.float32)
    sequence_length = np.array([3, 3], dtype=np.int32)
    merge_repeated = False
    blank_index = -1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    inputs = np.array([[[0.1, 0.6, 0.1, 0.2], [0.8, 0.1, 0.05, 0.05]],
                       [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]],
                       [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]], dtype=np.float32)
    sequence_length = np.array([3, 0], dtype=np.int32)
    merge_repeated = True
    blank_index = 3
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "merge_repeated": merge_repeated, "blank_index": blank_index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.ctc_greedy_decoder"] = tf_nn_ctc_greedy_decoder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.ctc_greedy_decoder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_greedy_decoder'.")

check_valid('tf.nn.ctc_greedy_decoder', generated_inputs['tf.nn.ctc_greedy_decoder'], lib="tf", suffix=0)
