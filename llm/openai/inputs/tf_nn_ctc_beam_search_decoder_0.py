
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.random.randn(4, 2, 3).astype(np.float32)
    sequence_length = np.array([4, 3], dtype=np.int32)
    beam_width = 3
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 2
    inputs = np.array([[[2.0, -1.0]]], dtype=np.float32)  # shape (1,1,2)
    sequence_length = np.array([1], dtype=np.int32)
    beam_width = 1
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 3
    inputs = (np.random.randn(6, 3, 5) * 2.0).astype(np.float32)
    sequence_length = np.array([6, 5, 4], dtype=np.int32)
    beam_width = 5
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 4
    inputs = np.random.uniform(-3, 3, size=(3, 3, 4)).astype(np.float32)
    sequence_length = np.array([3, 2, 2], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 5
    inputs = (np.random.randn(8, 2, 7) * 5.0 - 2.0).astype(np.float32)
    sequence_length = np.array([5, 8], dtype=np.int32)
    beam_width = 10
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 6
    inputs = (np.random.randn(10, 1, 10)).astype(np.float64)
    sequence_length = np.array([7], dtype=np.int32)
    beam_width = 50
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 7
    inputs = np.random.randn(2, 4, 3).astype(np.float32)
    sequence_length = np.array([2, 1, 2, 1], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 8
    inputs = (np.random.randn(5, 5, 6) * 0.1).astype(np.float32)
    sequence_length = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    beam_width = 3
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 9
    inputs = np.array([
        [[3.0, -1.0], [1.0, 0.0]],
        [[-2.0, 2.0], [0.5, -0.5]],
        [[0.1, -0.1], [1.5, -1.5]],
        [[-0.3, 0.3], [2.0, -2.0]],
    ], dtype=np.float32)  # shape (4,2,2)
    sequence_length = np.array([2, 4], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 10
    inputs = (np.random.randn(7, 3, 9)).astype(np.float64)
    sequence_length = np.array([7, 3, 6], dtype=np.int32)
    beam_width = 7
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    return list_of_inputs

generated_inputs["tf.nn.ctc_beam_search_decoder"] = tf_nn_ctc_beam_search_decoder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.ctc_beam_search_decoder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_beam_search_decoder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.ctc_beam_search_decoder', generated_inputs['tf.nn.ctc_beam_search_decoder'], lib="tf", suffix=0)
