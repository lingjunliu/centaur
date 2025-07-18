
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

@tf.keras.utils.register_keras_serializable()
class ShapefulSimpleRNNCell(tf.keras.layers.SimpleRNNCell):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.shape = (units, units)

@tf.keras.utils.register_keras_serializable()
class ShapefulLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.shape = (units, 4 * units)

@tf.keras.utils.register_keras_serializable()
class ShapefulGRUCell(tf.keras.layers.GRUCell):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.shape = (units, 3 * units)

def tf_nn_rnncelldropoutwrapper_inputs():
    list_of_inputs = []

    input_dict_1 = {
        'cell': ShapefulSimpleRNNCell(10),
        'input_keep_prob': 0.8,
        'output_keep_prob': 0.8,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 20,
        'dtype': 'float32',
        'seed': 123
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        'cell': ShapefulLSTMCell(5),
        'input_keep_prob': 0.5,
        'output_keep_prob': 1.0,
        'state_keep_prob': 0.7,
        'variational_recurrent': False,
        'input_size': 10,
        'dtype': 'float32',
        'seed': 456
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        'cell': ShapefulGRUCell(32),
        'input_keep_prob': 0.75,
        'output_keep_prob': 0.75,
        'state_keep_prob': 0.75,
        'variational_recurrent': True,
        'input_size': 64,
        'dtype': 'float64',
        'seed': 789
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        'cell': ShapefulSimpleRNNCell(16),
        'input_keep_prob': 1.0,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 16,
        'dtype': 'float64',
        'seed': 101
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {
        'cell': ShapefulLSTMCell(8),
        'input_keep_prob': 0.6,
        'output_keep_prob': 0.7,
        'state_keep_prob': 0.8,
        'variational_recurrent': False,
        'input_size': 12,
        'dtype': 'float32',
        'seed': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        'cell': ShapefulSimpleRNNCell(10),
        'input_keep_prob': 0.9,
        'output_keep_prob': 0.9,
        'state_keep_prob': 0.9,
        'variational_recurrent': False,
        'input_size': 10,
        'dtype': 'float32',
        'seed': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {
        'cell': ShapefulGRUCell(128),
        'input_keep_prob': 0.5,
        'output_keep_prob': 0.5,
        'state_keep_prob': 1.0,
        'variational_recurrent': True,
        'input_size': 256,
        'dtype': 'float32',
        'seed': 2023
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {
        'cell': ShapefulSimpleRNNCell(2),
        'input_keep_prob': 0.95,
        'output_keep_prob': 0.95,
        'state_keep_prob': 0.95,
        'variational_recurrent': False,
        'input_size': 4,
        'dtype': 'float32',
        'seed': 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_dict_9 = {
        'cell': ShapefulLSTMCell(40),
        'input_keep_prob': 0.8,
        'output_keep_prob': 1.0,
        'state_keep_prob': 0.8,
        'variational_recurrent': True,
        'input_size': 40,
        'dtype': 'float64',
        'seed': 1337
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {
        'cell': ShapefulSimpleRNNCell(64),
        'input_keep_prob': 0.7,
        'output_keep_prob': 0.6,
        'state_keep_prob': 0.5,
        'variational_recurrent': False,
        'input_size': 128,
        'dtype': 'float32',
        'seed': -100
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.RNNCellDropoutWrapper"] = tf_nn_rnncelldropoutwrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.RNNCellDropoutWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDropoutWrapper'.")

check_valid('tf.nn.RNNCellDropoutWrapper', generated_inputs['tf.nn.RNNCellDropoutWrapper'], lib="tf", suffix=0)
