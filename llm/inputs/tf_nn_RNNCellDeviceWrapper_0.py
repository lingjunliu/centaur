
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_rnncelldevicewrapper_inputs():
    list_of_inputs = []

    # Input 1
    cell = (tf.keras.layers.LSTMCell(units=10),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    cell = (tf.keras.layers.GRUCell(units=20),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    cell = (tf.keras.layers.SimpleRNNCell(units=30),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    cell = (tf.keras.layers.LSTMCell(units=5),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    cell = (tf.keras.layers.GRUCell(units=15),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    cell = (tf.keras.layers.LSTMCell(units=40),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    cell = (tf.keras.layers.SimpleRNNCell(units=60),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    cell = (tf.keras.layers.LSTMCell(units=70),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    cell = (tf.keras.layers.GRUCell(units=80),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    cell = (tf.keras.layers.SimpleRNNCell(units=90),)
    input_dict = {
        "cell": cell
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.RNNCellDeviceWrapper"] = tf_nn_rnncelldevicewrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.RNNCellDeviceWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDeviceWrapper'.")

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
