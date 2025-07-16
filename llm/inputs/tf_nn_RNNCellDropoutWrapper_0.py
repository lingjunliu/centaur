
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_rnncelldropoutwrapper_inputs():
    list_of_inputs = []

    # Input 1
    cell = tf.keras.layers.SimpleRNNCell(units=10)
    input_keep_prob = np.float32(0.8)
    output_keep_prob = np.float32(0.8)
    state_keep_prob = np.float32(0.8)
    variational_recurrent = np.bool_(False)
    input_size = np.int32(20)
    dtype = np.dtype(np.float32)
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
    cell = tf.keras.layers.LSTMCell(units=20)
    input_keep_prob = np.float64(0.6)
    output_keep_prob = np.float64(0.7)
    state_keep_prob = np.float64(0.5)
    variational_recurrent = np.bool_(True)
    input_size = np.int32(30)
    dtype = np.dtype(np.float64)
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
    cell = tf.keras.layers.GRUCell(units=5)
    input_keep_prob = np.float32(1.0)
    output_keep_prob = np.float32(1.0)
    state_keep_prob = np.float32(1.0)
    variational_recurrent = np.bool_(False)
    input_size = np.int32(10)
    dtype = np.dtype(np.float32)
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
    cell = tf.keras.layers.SimpleRNNCell(units=15)
    input_keep_prob = np.float64(0.9)
    output_keep_prob = np.float64(0.9)
    state_keep_prob = np.float64(0.9)
    variational_recurrent = np.bool_(True)
    input_size = np.int32(25)
    dtype = np.dtype(np.float64)
    seed = 101

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
    cell = tf.keras.layers.LSTMCell(units=8)
    input_keep_prob = np.float32(0.7)
    output_keep_prob = np.float32(0.6)
    state_keep_prob = np.float32(0.8)
    variational_recurrent = np.bool_(False)
    input_size = np.int32(12)
    dtype = np.dtype(np.float32)
    seed = 112

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
    cell = tf.keras.layers.GRUCell(units=12)
    input_keep_prob = np.float64(0.5)
    output_keep_prob = np.float64(0.4)
    state_keep_prob = np.float64(0.6)
    variational_recurrent = np.bool_(True)
    input_size = np.int32(18)
    dtype = np.dtype(np.float64)
    seed = 134

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
    cell = tf.keras.layers.SimpleRNNCell(units=7)
    input_keep_prob = np.float32(0.3)
    output_keep_prob = np.float32(0.2)
    state_keep_prob = np.float32(0.1)
    variational_recurrent = np.bool_(False)
    input_size = np.int32(9)
    dtype = np.dtype(np.float32)
    seed = 145

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
    cell = tf.keras.layers.LSTMCell(units=18)
    input_keep_prob = np.float64(0.2)
    output_keep_prob = np.float64(0.3)
    state_keep_prob = np.float64(0.4)
    variational_recurrent = np.bool_(True)
    input_size = np.int32(28)
    dtype = np.dtype(np.float64)
    seed = 156

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
    cell = tf.keras.layers.GRUCell(units=3)
    input_keep_prob = np.float32(0.0)
    output_keep_prob = np.float32(0.0)
    state_keep_prob = np.float32(0.0)
    variational_recurrent = np.bool_(False)
    input_size = np.int32(4)
    dtype = np.dtype(np.float32)
    seed = 167

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
    cell = tf.keras.layers.SimpleRNNCell(units=22)
    input_keep_prob = np.float64(0.4)
    output_keep_prob = np.float64(0.5)
    state_keep_prob = np.float64(0.3)
    variational_recurrent = np.bool_(True)
    input_size = np.int32(32)
    dtype = np.dtype(np.float64)
    seed = 178

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
generated_inputs["tf.nn.RNNCellDropoutWrapper"] = tf_nn_rnncelldropoutwrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.RNNCellDropoutWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDropoutWrapper'.")

check_valid('tf.nn.RNNCellDropoutWrapper', generated_inputs['tf.nn.RNNCellDropoutWrapper'], lib="tf", suffix=0)
