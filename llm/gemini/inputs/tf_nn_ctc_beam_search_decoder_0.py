
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.random.rand(5, 1, 10).astype(np.float32)
    sequence_length = np.array([5]).astype(np.int32)
    beam_width = 10
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.random.rand(10, 2, 20).astype(np.float32)
    sequence_length = np.array([10, 8]).astype(np.int32)
    beam_width = 50
    top_paths = 5
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.random.rand(15, 3, 5).astype(np.float32)
    sequence_length = np.array([15, 12, 10]).astype(np.int32)
    beam_width = 100
    top_paths = 10
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.random.rand(20, 4, 15).astype(np.float32)
    sequence_length = np.array([20, 18, 15, 12]).astype(np.int32)
    beam_width = 25
    top_paths = 3
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.random.rand(3, 1, 2).astype(np.float32)
    sequence_length = np.array([3]).astype(np.int32)
    beam_width = 1
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.random.rand(7, 2, 8).astype(np.float32)
    sequence_length = np.array([7, 5]).astype(np.int32)
    beam_width = 2
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.random.rand(12, 3, 12).astype(np.float32)
    sequence_length = np.array([12, 9, 6]).astype(np.int32)
    beam_width = 15
    top_paths = 2
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    inputs = np.random.rand(8, 1, 4).astype(np.float32)
    sequence_length = np.array([8]).astype(np.int32)
    beam_width = 1
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.random.rand(6, 2, 6).astype(np.float32)
    sequence_length = np.array([6, 4]).astype(np.int32)
    beam_width = 4
    top_paths = 2
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.random.rand(4, 4, 3).astype(np.float32)
    sequence_length = np.array([4, 3, 2, 1]).astype(np.int32)
    beam_width = 3
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    inputs = np.random.rand(1, 1, 1).astype(np.float32)
    sequence_length = np.array([1]).astype(np.int32)
    beam_width = 1
    top_paths = 1
    input_dict = {"inputs": inputs, "sequence_length": sequence_length, "beam_width": beam_width, "top_paths": top_paths}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.ctc_beam_search_decoder"] = tf_nn_ctc_beam_search_decoder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.ctc_beam_search_decoder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_beam_search_decoder'.")

check_valid('tf.nn.ctc_beam_search_decoder', generated_inputs['tf.nn.ctc_beam_search_decoder'], lib="tf", suffix=0)
