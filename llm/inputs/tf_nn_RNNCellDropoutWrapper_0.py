
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDropoutWrapper_inputs():
    list_of_inputs = []

    # Input 1
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=10, dtype=tf.float32)
    input_keep_prob = 0.5
    output_keep_prob = 0.7
    state_keep_prob = 1.0
    variational_recurrent = False
    input_size = 5
    dtype = tf.float32
    seed = 123

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
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=20, dtype=tf.float64)
    input_keep_prob = 0.8
    output_keep_prob = 0.9
    state_keep_prob = 0.6
    variational_recurrent = True
    input_size = 10
    dtype = tf.float64
    seed = 456

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
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=15, dtype=tf.float32)
    input_keep_prob = 1.0
    output_keep_prob = 1.0
    state_keep_prob = 1.0
    variational_recurrent = False
    input_size = 8
    dtype = tf.float32
    seed = 789

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
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=8, dtype=tf.float64)
    input_keep_prob = 0.3
    output_keep_prob = 0.4
    state_keep_prob = 0.5
    variational_recurrent = True
    input_size = 4
    dtype = tf.float64
    seed = 987

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
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=12, dtype=tf.float32)
    input_keep_prob = 0.6
    output_keep_prob = 0.7
    state_keep_prob = 0.8
    variational_recurrent = False
    input_size = 6
    dtype = tf.float32
    seed = 654

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
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=16, dtype=tf.float64)
    input_keep_prob = 0.9
    output_keep_prob = 0.2
    state_keep_prob = 0.3
    variational_recurrent = True
    input_size = 7
    dtype = tf.float64
    seed = 321

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
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=32, dtype=tf.float32)
    input_keep_prob = 0.1
    output_keep_prob = 0.95
    state_keep_prob = 0.55
    variational_recurrent = False
    input_size = 32
    dtype = tf.float32
    seed = 159

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
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=30, dtype=tf.float64)
    input_keep_prob = 0.25
    output_keep_prob = 0.65
    state_keep_prob = 0.95
    variational_recurrent = True
    input_size = 15
    dtype = tf.float64
    seed = 753

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
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=25, dtype=tf.float32)
    input_keep_prob = 0.75
    output_keep_prob = 0.35
    state_keep_prob = 0.05
    variational_recurrent = False
    input_size = 12
    dtype = tf.float32
    seed = 951

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
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=5, dtype=tf.float64)
    input_keep_prob = 0.01
    output_keep_prob = 0.01
    state_keep_prob = 0.01
    variational_recurrent = True
    input_size = 1
    dtype = tf.float64
    seed = 100

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
