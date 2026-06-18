
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import tensorflow as tf


def tf_nn_ctc_greedy_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.random.randn(5, 2, 3).astype(np.float32)
    sequence_length = np.array([5, 4], dtype=np.int32)
    merge_repeated = True
    blank_index = 2

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.random.randn(10, 3, 5).astype(np.float32)
    sequence_length = np.array([10, 8, 9], dtype=np.int32)
    merge_repeated = False
    blank_index = 0

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.random.randn(2, 1, 2).astype(np.float32)
    sequence_length = np.array([2], dtype=np.int32)
    merge_repeated = True
    blank_index = -1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.random.randn(15, 4, 10).astype(np.float32)
    sequence_length = np.array([12, 15, 10, 14], dtype=np.int32)
    merge_repeated = False
    blank_index = 9

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.random.randn(8, 2, 4).astype(np.float32)
    sequence_length = np.array([8, 8], dtype=np.int32)
    merge_repeated = True
    blank_index = 3

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.random.randn(6, 2, 3).astype(np.float32)
    sequence_length = np.array([6, 5], dtype=np.int32)
    merge_repeated = True
    blank_index = 1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.random.randn(4, 5, 3).astype(np.float32)
    sequence_length = np.array([4, 3, 2, 4, 1], dtype=np.int32)
    merge_repeated = False
    blank_index = 2

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.random.randn(3, 3, 3).astype(np.float32)
    sequence_length = np.array([3, 2, 1], dtype=np.int32)
    merge_repeated = True
    blank_index = -1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.random.randn(20, 1, 50).astype(np.float32)
    sequence_length = np.array([20], dtype=np.int32)
    merge_repeated = False
    blank_index = 49

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.random.randn(7, 3, 6).astype(np.float32)
    sequence_length = np.array([5, 6, 7], dtype=np.int32)
    merge_repeated = True
    blank_index = 4

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.nn.ctc_greedy_decoder"] = tf_nn_ctc_greedy_decoder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.ctc_greedy_decoder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_greedy_decoder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.ctc_greedy_decoder', generated_inputs['tf.nn.ctc_greedy_decoder'], lib="tf", suffix=0)
