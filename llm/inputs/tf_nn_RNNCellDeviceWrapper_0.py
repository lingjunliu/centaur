
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDeviceWrapper_inputs():
    list_of_inputs = []

    # Input 1: SimpleRNNCell
    cell_1 = tf.keras.layers.SimpleRNNCell(units=128)
    input_dict_1 = {"cell": (cell_1,)}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: LSTMCell
    cell_2 = tf.keras.layers.LSTMCell(units=256)
    input_dict_2 = {"cell": (cell_2,)}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: GRUCell
    cell_3 = tf.keras.layers.GRUCell(units=64)
    input_dict_3 = {"cell": (cell_3,)}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4:  LSTM
    cell_4 = tf.keras.layers.LSTMCell(units=32)
    input_dict_4 = {"cell": (cell_4,)}
    list_of_inputs.append(copy.deepcopy(input_dict_4))


    # Input 5: Using LSTMCell again
    base_cell_5 = tf.keras.layers.LSTMCell(units=128)
    input_dict_5 = {"cell": (base_cell_5,)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6:  Trying SimpleRNN with a different unit size.
    base_cell_6 = tf.keras.layers.SimpleRNNCell(units=200)
    input_dict_6 = {"cell": (base_cell_6,)}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Gated Recurrent Unit
    cell_7 = tf.keras.layers.GRUCell(units=128)
    input_dict_7 = {"cell": (cell_7,)}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8:  More LSTM
    cell_8 = tf.keras.layers.LSTMCell(units=256)
    input_dict_8 = {"cell": (cell_8,)}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Some GRU
    cell_9 = tf.keras.layers.GRUCell(units=64)
    input_dict_9 = {"cell": (cell_9,)}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: SimpleRNN
    cell_10 = tf.keras.layers.SimpleRNNCell(units=128)
    input_dict_10 = {"cell": (cell_10,)}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.RNNCellDeviceWrapper"] = tf_nn_RNNCellDeviceWrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.RNNCellDeviceWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDeviceWrapper'.")

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
