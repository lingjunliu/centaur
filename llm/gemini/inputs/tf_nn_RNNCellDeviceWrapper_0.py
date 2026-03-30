
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDeviceWrapper_inputs():
    list_of_inputs = []

    # Input 1: A basic LSTM cell
    cell1 = tf.keras.layers.LSTMCell(units=32)
    input_dict1 = {"cell": (cell1,)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: A basic GRU cell
    cell2 = tf.keras.layers.GRUCell(units=64)
    input_dict2 = {"cell": (cell2,)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: A basic RNN cell
    cell3 = tf.keras.layers.SimpleRNNCell(units=128)
    input_dict3 = {"cell": (cell3,)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: A stacked LSTM cell
    cell4_1 = tf.keras.layers.LSTMCell(units=16)
    cell4_2 = tf.keras.layers.LSTMCell(units=32)
    cell4 = tf.keras.layers.StackedRNNCells([cell4_1, cell4_2])
    input_dict4 = {"cell": (cell4,)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Another LSTM cell with different params
    cell5 = tf.keras.layers.LSTMCell(units=8, activation='relu', recurrent_activation='sigmoid', use_bias=False)
    input_dict5 = {"cell": (cell5,)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: GRU cell, another configuration
    cell6 = tf.keras.layers.GRUCell(units=24, activation='tanh', recurrent_activation='sigmoid')
    input_dict6 = {"cell": (cell6,)}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: A basic RNN cell with a different activation.
    cell7 = tf.keras.layers.SimpleRNNCell(units=64, activation='relu')
    input_dict7 = {"cell": (cell7,)}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: A LSTM Cell with dropout
    cell8 = tf.keras.layers.LSTMCell(units=32, dropout=0.2, recurrent_dropout=0.2)
    input_dict8 = {"cell": (cell8,)}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: GRU Cell with dropout
    cell9 = tf.keras.layers.GRUCell(units=16, dropout=0.5, recurrent_dropout=0.5)
    input_dict9 = {"cell": (cell9,)}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: A complicated cell.
    cell10_1 = tf.keras.layers.LSTMCell(units=8)
    cell10_2 = tf.keras.layers.GRUCell(units=16)
    cell10 = tf.keras.layers.StackedRNNCells([cell10_1, cell10_2])
    input_dict10 = {"cell": (cell10,)}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.RNNCellDeviceWrapper"] = tf_nn_RNNCellDeviceWrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.RNNCellDeviceWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDeviceWrapper'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
