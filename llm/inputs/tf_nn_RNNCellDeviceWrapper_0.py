
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_rnn_cell_device_wrapper_inputs():
    list_of_inputs = []

    # Input 1: BasicRNNCell
    cell1 = tf.keras.layers.SimpleRNNCell(units=32)
    input_dict1 = {"cell": (cell1,)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: LSTMCell
    cell2 = tf.keras.layers.LSTMCell(units=64)
    input_dict2 = {"cell": (cell2,)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: GRUCell
    cell3 = tf.keras.layers.GRUCell(units=128)
    input_dict3 = {"cell": (cell3,)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: LSTMCell
    cell4 = tf.keras.layers.LSTMCell(units=256)
    input_dict4 = {"cell": (cell4,)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: SimpleRNNCell
    cell5 = tf.keras.layers.SimpleRNNCell(units=16)
    input_dict5 = {"cell": (cell5,)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: GRUCell
    cell6 = tf.keras.layers.GRUCell(units=64)
    input_dict6 = {"cell": (cell6,)}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: LSTMCell
    cell7 = tf.keras.layers.LSTMCell(units=128)
    input_dict7 = {"cell": (cell7,)}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: SimpleRNNCell
    cell8 = tf.keras.layers.SimpleRNNCell(units=32)
    input_dict8 = {"cell": (cell8,)}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: GRUCell
    cell9 = tf.keras.layers.GRUCell(units=16)
    input_dict9 = {"cell": (cell9,)}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: LSTMCell
    cell10 = tf.keras.layers.LSTMCell(units=32)
    input_dict10 = {"cell": (cell10,)}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.RNNCellDeviceWrapper"] = tf_nn_rnn_cell_device_wrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.RNNCellDeviceWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDeviceWrapper'.")

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
