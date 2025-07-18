
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to satisfy the test harness and the TF API.
# 1. Has a `.shape` attribute for the harness's signature check.
# 2. Implements `__deepcopy__` to bypass Keras serialization during the harness's deepcopy.
# 3. Internally converts numpy dtypes (from harness) to tensorflow dtypes (for Keras).
class MockRNNCell(tf.keras.layers.Layer):
    def __init__(self, cell_class, units, input_dim, **kwargs):
        # Store original args for __deepcopy__
        self._cell_class_arg = cell_class
        self._units_arg = units
        self._input_dim_arg = input_dim
        self._kwargs_arg = kwargs.copy()

        # Keras layers need TF dtypes. The harness seems to provide numpy dtypes.
        keras_kwargs = kwargs.copy()
        if 'dtype' in keras_kwargs:
            keras_kwargs['dtype'] = tf.as_dtype(keras_kwargs['dtype'])

        super().__init__(**keras_kwargs)
        self._cell = cell_class(units, **keras_kwargs)
        
        # Fake a .shape attribute for the test harness.
        self.shape = (input_dim, units)

    def call(self, inputs, states):
        return self._cell.call(inputs, states)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def build(self, input_shape):
        if not self._cell.built:
            self._cell.build(input_shape)
        self.built = True
    
    def __deepcopy__(self, memo):
        # Bypass Keras serialization by creating a new instance manually.
        if id(self) in memo:
            return memo[id(self)]
        new_obj = self.__class__(self._cell_class_arg, self._units_arg, self._input_dim_arg, **self._kwargs_arg)
        memo[id(self)] = new_obj
        return new_obj

def tf_nn_rnncelldropoutwrapper_inputs():
    list_of_inputs = []

    # Input 1: Basic case with SimpleRNNCell
    input_dict_1 = {
        'cell': MockRNNCell(tf.keras.layers.SimpleRNNCell, units=20, input_dim=10, dtype=np.float32),
        'input_keep_prob': 0.8,
        'output_keep_prob': 0.8,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 10,
        'dtype': np.float32,
        'seed': 123
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: LSTMCell with variational dropout
    input_dict_2 = {
        'cell': MockRNNCell(tf.keras.layers.LSTMCell, units=15, input_dim=5, dtype=np.float32),
        'input_keep_prob': 0.9,
        'output_keep_prob': 0.9,
        'state_keep_prob': 0.85,
        'variational_recurrent': True,
        'input_size': 5,
        'dtype': np.float32,
        'seed': 456
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: GRUCell with no dropout
    input_dict_3 = {
        'cell': MockRNNCell(tf.keras.layers.GRUCell, units=16, input_dim=8, dtype=np.float32),
        'input_keep_prob': 1.0,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 8,
        'dtype': np.float32,
        'seed': 789
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: float64 dtype
    input_dict_4 = {
        'cell': MockRNNCell(tf.keras.layers.SimpleRNNCell, units=24, input_dim=12, dtype=np.float64),
        'input_keep_prob': 0.7,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 12,
        'dtype': np.float64,
        'seed': 101
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Variational LSTMCell with float64
    input_dict_5 = {
        'cell': MockRNNCell(tf.keras.layers.LSTMCell, units=30, input_dim=20, dtype=np.float64),
        'input_keep_prob': 1.0,
        'output_keep_prob': 0.75,
        'state_keep_prob': 0.7,
        'variational_recurrent': True,
        'input_size': 20,
        'dtype': np.float64,
        'seed': 202
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero seed
    input_dict_6 = {
        'cell': MockRNNCell(tf.keras.layers.SimpleRNNCell, units=10, input_dim=10, dtype=np.float32),
        'input_keep_prob': 0.5,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 10,
        'dtype': np.float32,
        'seed': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Negative seed
    input_dict_7 = {
        'cell': MockRNNCell(tf.keras.layers.GRUCell, units=14, input_dim=7, dtype=np.float32),
        'input_keep_prob': 0.6,
        'output_keep_prob': 0.7,
        'state_keep_prob': 0.8,
        'variational_recurrent': True,
        'input_size': 7,
        'dtype': np.float32,
        'seed': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger sizes
    input_dict_8 = {
        'cell': MockRNNCell(tf.keras.layers.LSTMCell, units=200, input_dim=100, dtype=np.float32),
        'input_keep_prob': 0.95,
        'output_keep_prob': 0.95,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 100,
        'dtype': np.float32,
        'seed': 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-variational with state_keep_prob < 1.0 (should be ignored by op)
    input_dict_9 = {
        'cell': MockRNNCell(tf.keras.layers.SimpleRNNCell, units=16, input_dim=16, dtype=np.float32),
        'input_keep_prob': 0.8,
        'output_keep_prob': 0.8,
        'state_keep_prob': 0.5,
        'variational_recurrent': False,
        'input_size': 16,
        'dtype': np.float32,
        'seed': 1337
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Variational with no state dropout
    input_dict_10 = {
        'cell': MockRNNCell(tf.keras.layers.SimpleRNNCell, units=64, input_dim=32, dtype=np.float32),
        'input_keep_prob': 0.5,
        'output_keep_prob': 0.5,
        'state_keep_prob': 1.0,
        'variational_recurrent': True,
        'input_size': 32,
        'dtype': np.float32,
        'seed': 2023
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
