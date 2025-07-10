
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDropoutWrapper_inputs():
    list_of_inputs = []

    # Input 1
    cell = tf.keras.layers.SimpleRNNCell(units=32)
    input_keep_prob = 0.8
    output_keep_prob = 0.8
    state_keep_prob = 0.8
    variational_recurrent = True
    input_size = 64
    dtype = tf.float32
    seed = 123

    input_dict = {
        "cell": cell,
        "input_keep_prob": np.float32(input_keep_prob),
        "output_keep_prob": np.float32(output_keep_prob),
        "state_keep_prob": np.float32(state_keep_prob),
        "variational_recurrent": np.bool_(variational_recurrent),
        "input_size": np.int32(input_size),
        "dtype": dtype,
        "seed": np.int32(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.RNNCellDropoutWrapper"] = tf_nn_RNNCellDropoutWrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.RNNCellDropoutWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDropoutWrapper'.")

check_valid('tf.nn.RNNCellDropoutWrapper', generated_inputs['tf.nn.RNNCellDropoutWrapper'], lib="tf", suffix=0)
