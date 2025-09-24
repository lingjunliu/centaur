
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_RNNCellDeviceWrapper_inputs():
    list_of_inputs = []

    # The series of errors indicates a complex interaction between the test harness
    # constraints and the API's signature.
    # 1. `TypeError: ... missing ... 'device'` means the API needs a cell and a device.
    # 2. `ValueError: setting an array element with a sequence` means the test harness
    #    requires all inputs for the 'cell' key to be flat, numeric tuples of the
    #    same length so they can be converted to a homogeneous numpy array.
    #
    # The solution is to encode all required information (cell type, cell arguments,
    # and device) into a single, flat, fixed-length numeric tuple.
    #
    # Proposed fixed-length tuple structure:
    # (cell_type, units, use_peepholes, forget_bias, device_code)
    # - cell_type: 1=BasicRNN, 2=GRU, 3=LSTM
    # - units: Integer
    # - use_peepholes: 0=False, 1=True. (Placeholder 0 for non-LSTM cells)
    # - forget_bias: Float. (Placeholder 0.0 for non-LSTM cells)
    # - device_code: 0=/cpu:0, 1=/gpu:0, 2=/gpu:1

    # Input 1: BasicRNNCell(64) on CPU
    input_dict = {'cell': (1, 64, 0, 0.0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: GRUCell(128) on GPU
    input_dict = {'cell': (2, 128, 0, 0.0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: LSTMCell(256) on CPU (using default peepholes=False, forget_bias=1.0)
    input_dict = {'cell': (3, 256, 0, 1.0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: BasicRNNCell(32) on GPU
    input_dict = {'cell': (1, 32, 0, 0.0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: GRUCell(512) on CPU
    input_dict = {'cell': (2, 512, 0, 0.0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large LSTMCell(1024) on GPU
    input_dict = {'cell': (3, 1024, 0, 1.0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: LSTMCell(128) with peepholes enabled on CPU
    input_dict = {'cell': (3, 128, 1, 1.0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: LSTMCell(256) with a custom forget_bias on GPU
    input_dict = {'cell': (3, 256, 0, 1.5, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: LSTMCell(64) with peepholes and a negative forget_bias on CPU
    input_dict = {'cell': (3, 64, 1, -1.0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small GRUCell(16) on another GPU device
    input_dict = {'cell': (2, 16, 0, 0.0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Smallest LSTMCell(8) with peepholes and zero forget_bias on GPU
    input_dict = {'cell': (3, 8, 1, 0.0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

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

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
