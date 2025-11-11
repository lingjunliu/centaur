
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    inputs1 = np.random.rand(10, 4, 5).astype(np.float32)
    sequence_length1 = np.array([4, 4, 4, 4], dtype=np.int32)
    beam_width1 = 10
    top_paths1 = 2

    input_dict1 = {
        "inputs": inputs1,
        "sequence_length": sequence_length1,
        "beam_width": beam_width1,
        "top_paths": top_paths1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

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
