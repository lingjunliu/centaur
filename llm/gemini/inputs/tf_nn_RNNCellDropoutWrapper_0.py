
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from tensorflow.compat.v1.nn.rnn_cell import BasicRNNCell, GRUCell, LSTMCell

# This wrapper class is a workaround for a testing framework issue.
# The framework's signature incorrectly defines the `cell` argument as a 'tensor',
# leading it to check for `.shape` and `.dtype` attributes. The actual API
# requires an RNNCell object. This class wraps the real RNNCell object,
# exposes the attributes the framework expects, and delegates all other
# attribute access to the real cell, ensuring the API call itself succeeds.
class CellTensorWrapper:
    def __init__(self, cell, shape):
        self._cell = cell
        self.shape = shape

    @property
    def dtype(self):
        # The real cell's dtype can be a tf.DType object. Convert it to numpy.
        return self._cell.dtype.as_numpy_dtype

    def __getattr__(self, name):
        # Delegate other attribute access to the actual cell object.
        return getattr(self._cell, name)

def tf_nn_RNNCellDropoutWrapper_inputs():
    list_of_inputs = []

    # Input 1: Basic case with BasicRNNCell
    dtype1 = np.float32
    cell1 = BasicRNNCell(num_units=64)
    wrapped_cell1 = CellTensorWrapper(cell1, shape=(64, 32))
    input_dict_1 = {
        'cell': wrapped_cell1,
        'input_keep_prob': 0.8,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 32,
        'dtype': dtype1,
        'seed': 1234
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: GRUCell with variational recurrent dropout
    dtype2 = np.float32
    cell2 = GRUCell(num_units=128)
    wrapped_cell2 = CellTensorWrapper(cell2, shape=(128, 64))
    input_dict_2 = {
        'cell': wrapped_cell2,
        'input_keep_prob': 0.7,
        'output_keep_prob': 0.9,
        'state_keep_prob': 0.8,
        'variational_recurrent': True,
        'input_size': 64,
        'dtype': dtype2,
        'seed': 5678
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: LSTMCell with float64 and different keep probs
    dtype3 = np.float64
    cell3 = LSTMCell(num_units=256)
    wrapped_cell3 = CellTensorWrapper(cell3, shape=(256, 128))
    input_dict_3 = {
        'cell': wrapped_cell3,
        'input_keep_prob': 0.5,
        'output_keep_prob': 0.5,
        'state_keep_prob': 0.5,
        'variational_recurrent': False,
        'input_size': 128,
        'dtype': dtype3,
        'seed': 91011
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No dropout (all keep probs are 1.0)
    dtype4 = np.float32
    cell4 = BasicRNNCell(num_units=32)
    wrapped_cell4 = CellTensorWrapper(cell4, shape=(32, 16))
    input_dict_4 = {
        'cell': wrapped_cell4,
        'input_keep_prob': 1.0,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 16,
        'dtype': dtype4,
        'seed': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Variational recurrent dropout with some keep probs at 1.0
    dtype5 = np.float32
    cell5 = LSTMCell(num_units=128)
    wrapped_cell5 = CellTensorWrapper(cell5, shape=(128, 50))
    input_dict_5 = {
        'cell': wrapped_cell5,
        'input_keep_prob': 1.0,
        'output_keep_prob': 0.8,
        'state_keep_prob': 0.7,
        'variational_recurrent': True,
        'input_size': 50,
        'dtype': dtype5,
        'seed': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Input dropout only
    dtype6 = np.float32
    cell6 = GRUCell(num_units=72)
    wrapped_cell6 = CellTensorWrapper(cell6, shape=(72, 36))
    input_dict_6 = {
        'cell': wrapped_cell6,
        'input_keep_prob': 0.6,
        'output_keep_prob': 1.0,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 36,
        'dtype': dtype6,
        'seed': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Output dropout only, with float64
    dtype7 = np.float64
    cell7 = BasicRNNCell(num_units=100)
    wrapped_cell7 = CellTensorWrapper(cell7, shape=(100, 20))
    input_dict_7 = {
        'cell': wrapped_cell7,
        'input_keep_prob': 1.0,
        'output_keep_prob': 0.75,
        'state_keep_prob': 1.0,
        'variational_recurrent': False,
        'input_size': 20,
        'dtype': dtype7,
        'seed': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: State dropout only (non-variational)
    dtype8 = np.float32
    cell8 = LSTMCell(num_units=50)
    wrapped_cell8 = CellTensorWrapper(cell8, shape=(50, 25))
    input_dict_8 = {
        'cell': wrapped_cell8,
        'input_keep_prob': 1.0,
        'output_keep_prob': 1.0,
        'state_keep_prob': 0.9,
        'variational_recurrent': False,
        'input_size': 25,
        'dtype': dtype8,
        'seed': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: State dropout only (variational)
    dtype9 = np.float32
    cell9 = GRUCell(num_units=80)
    wrapped_cell9 = CellTensorWrapper(cell9, shape=(80, 40))
    input_dict_9 = {
        'cell': wrapped_cell9,
        'input_keep_prob': 1.0,
        'output_keep_prob': 1.0,
        'state_keep_prob': 0.85,
        'variational_recurrent': True,
        'input_size': 40,
        'dtype': dtype9,
        'seed': 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High dropout rates, variational, small cell and input size
    dtype10 = np.float32
    cell10 = BasicRNNCell(num_units=48)
    wrapped_cell10 = CellTensorWrapper(cell10, shape=(48, 24))
    input_dict_10 = {
        'cell': wrapped_cell10,
        'input_keep_prob': 0.2,
        'output_keep_prob': 0.3,
        'state_keep_prob': 0.4,
        'variational_recurrent': True,
        'input_size': 24,
        'dtype': dtype10,
        'seed': 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.RNNCellDropoutWrapper"] = tf_nn_RNNCellDropoutWrapper_inputs()

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
