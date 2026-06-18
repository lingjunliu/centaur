
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    # Input 1: Standard small configuration
    inputs = np.random.randn(5, 2, 3).astype(np.float32)
    seq_len = np.array([5, 4], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 10,
        'top_paths': 2
    })

    # Input 2: Minimal configuration
    inputs = np.random.randn(1, 1, 2).astype(np.float32)
    seq_len = np.array([1], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 1,
        'top_paths': 1
    })

    # Input 3: Medium configuration with beam_width equal to top_paths
    inputs = np.random.randn(10, 4, 8).astype(np.float32)
    seq_len = np.array([8, 10, 5, 9], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 5,
        'top_paths': 5
    })

    # Input 4: Large number of classes with float64 inputs
    inputs = np.random.randn(12, 3, 50).astype(np.float64)
    seq_len = np.array([12, 10, 11], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 20,
        'top_paths': 3
    })

    # Input 5: High beam width and multiple top paths
    inputs = np.random.randn(8, 2, 10).astype(np.float32)
    seq_len = np.array([6, 8], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 100,
        'top_paths': 10
    })

    # Input 6: Variable small sequence lengths
    inputs = np.random.randn(15, 5, 6).astype(np.float32)
    seq_len = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 8,
        'top_paths': 4
    })

    # Input 7: Larger batch size with uniform sequence length
    inputs = np.random.randn(6, 16, 4).astype(np.float32)
    seq_len = np.array([6] * 16, dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 4,
        'top_paths': 2
    })

    # Input 8: Hand-crafted deterministic values
    inputs = np.ones((4, 2, 5), dtype=np.float32) * -1.5
    inputs[0, 0, 1] = 2.0
    inputs[1, 1, 2] = 3.5
    seq_len = np.array([4, 3], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 3,
        'top_paths': 1
    })

    # Input 9: Uniformly distributed inputs
    inputs = np.random.uniform(-10.0, 10.0, (7, 3, 12)).astype(np.float32)
    seq_len = np.array([7, 6, 5], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 12,
        'top_paths': 6
    })

    # Input 10: Single top path with substantial beam width
    inputs = np.random.randn(20, 2, 15).astype(np.float32)
    seq_len = np.array([20, 18], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 50,
        'top_paths': 1
    })

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
