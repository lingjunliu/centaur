
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_hann_window_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'window_length': np.array(8, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float32,
        'name': "hann_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'window_length': np.array(16, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float64,
        'name': "hann_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'window_length': np.array(1, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float16,
        'name': "hann_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'window_length': np.array(64, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float32,
        'name': "hann_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'window_length': np.array(128, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float64,
        'name': "hann_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'window_length': np.array(3, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float16,
        'name': "hann_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'window_length': np.array(1024, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float32,
        'name': "hann_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'window_length': np.array(256, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float32,
        'name': "hann_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'window_length': np.array(5, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float64,
        'name': "hann_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'window_length': np.array(512, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float64,
        'name': "hann_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.signal.hann_window"] = tf_signal_hann_window_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.hann_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.hann_window'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.hann_window', generated_inputs['tf.signal.hann_window'], lib="tf", suffix=0)
