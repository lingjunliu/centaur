
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDropoutWrapper_inputs():
    list_of_inputs = []

    # Input 1
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(10)
    input_keep_prob = np.float32(0.5)
    output_keep_prob = np.float32(0.7)
    state_keep_prob = np.float32(1.0)
    variational_recurrent = False
    input_size = np.int32(5)
    dtype = np.float32
    seed = np.int32(123)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(20)
    input_keep_prob = np.float32(0.8)
    output_keep_prob = np.float32(0.9)
    state_keep_prob = np.float32(0.95)
    variational_recurrent = True
    input_size = np.int32(10)
    dtype = np.float64
    seed = np.int32(456)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(15)
    input_keep_prob = np.float32(0.6)
    output_keep_prob = np.float32(0.65)
    state_keep_prob = np.float32(0.7)
    variational_recurrent = False
    input_size = np.int32(8)
    dtype = np.float16
    seed = np.int32(789)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(5)
    input_keep_prob = np.float32(0.9)
    output_keep_prob = np.float32(0.95)
    state_keep_prob = np.float32(1.0)
    variational_recurrent = True
    input_size = np.int32(3)
    dtype = np.float32
    seed = np.int32(101)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(32)
    input_keep_prob = np.float32(1.0)
    output_keep_prob = np.float32(1.0)
    state_keep_prob = np.float32(1.0)
    variational_recurrent = False
    input_size = np.int32(16)
    dtype = np.float64
    seed = np.int32(202)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(8)
    input_keep_prob = np.float32(0.4)
    output_keep_prob = np.float32(0.5)
    state_keep_prob = np.float32(0.6)
    variational_recurrent = True
    input_size = np.int32(4)
    dtype = np.float16
    seed = np.int32(303)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(64)
    input_keep_prob = np.float32(0.75)
    output_keep_prob = np.float32(0.85)
    state_keep_prob = np.float32(0.9)
    variational_recurrent = False
    input_size = np.int32(32)
    dtype = np.float32
    seed = np.int32(404)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(128)
    input_keep_prob = np.float32(0.25)
    output_keep_prob = np.float32(0.35)
    state_keep_prob = np.float32(0.45)
    variational_recurrent = True
    input_size = np.int32(64)
    dtype = np.float64
    seed = np.int32(505)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(4)
    input_keep_prob = np.float32(0.99)
    output_keep_prob = np.float32(0.98)
    state_keep_prob = np.float32(0.97)
    variational_recurrent = False
    input_size = np.int32(2)
    dtype = np.float16
    seed = np.int32(606)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(16)
    input_keep_prob = np.float32(0.1)
    output_keep_prob = np.float32(0.2)
    state_keep_prob = np.float32(0.3)
    variational_recurrent = True
    input_size = np.int32(1)
    dtype = np.float32
    seed = np.int32(707)

    input_dict = {
        "cell": cell,
        "input_keep_prob": input_keep_prob,
        "output_keep_prob": output_keep_prob,
        "state_keep_prob": state_keep_prob,
        "variational_recurrent": variational_recurrent,
        "input_size": input_size,
        "dtype": dtype,
        "seed": seed
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
