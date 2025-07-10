
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_linear_to_mel_weight_matrix_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "num_mel_bins": 20,
        "num_spectrogram_bins": 129,
        "sample_rate": 8000.0,
        "lower_edge_hertz": 125.0,
        "upper_edge_hertz": 3800.0,
        "dtype": tf.float32,
        "name": "mel_matrix_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "num_mel_bins": 40,
        "num_spectrogram_bins": 257,
        "sample_rate": 16000.0,
        "lower_edge_hertz": 0.0,
        "upper_edge_hertz": 8000.0,
        "dtype": tf.float64,
        "name": "mel_matrix_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "num_mel_bins": 128,
        "num_spectrogram_bins": 513,
        "sample_rate": 22050.0,
        "lower_edge_hertz": 50.0,
        "upper_edge_hertz": 11025.0,
        "dtype": tf.float32,
        "name": "mel_matrix_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "num_mel_bins": 64,
        "num_spectrogram_bins": 1025,
        "sample_rate": 44100.0,
        "lower_edge_hertz": 100.0,
        "upper_edge_hertz": 22050.0,
        "dtype": tf.float64,
        "name": "mel_matrix_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_dict = {
        "num_mel_bins": 32,
        "num_spectrogram_bins": 65,
        "sample_rate": 8000.0,
        "lower_edge_hertz": 500.0,
        "upper_edge_hertz": 3500.0,
        "dtype": tf.float32,
        "name": "mel_matrix_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "num_mel_bins": 80,
        "num_spectrogram_bins": 4097,
        "sample_rate": 48000.0,
        "lower_edge_hertz": 75.0,
        "upper_edge_hertz": 24000.0,
        "dtype": tf.float64,
        "name": "mel_matrix_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "num_mel_bins": 24,
        "num_spectrogram_bins": 33,
        "sample_rate": 4000.0,
        "lower_edge_hertz": 200.0,
        "upper_edge_hertz": 1800.0,
        "dtype": tf.float32,
        "name": "mel_matrix_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "num_mel_bins": 96,
        "num_spectrogram_bins": 1025,
        "sample_rate": 44100.0,
        "lower_edge_hertz": 20.0,
        "upper_edge_hertz": 20000.0,
        "dtype": tf.float64,
        "name": "mel_matrix_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "num_mel_bins": 16,
        "num_spectrogram_bins": 17,
        "sample_rate": 2000.0,
        "lower_edge_hertz": 300.0,
        "upper_edge_hertz": 900.0,
        "dtype": tf.float32,
        "name": "mel_matrix_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "num_mel_bins": 100,
        "num_spectrogram_bins": 2049,
        "sample_rate": 48000.0,
        "lower_edge_hertz": 10.0,
        "upper_edge_hertz": 23000.0,
        "dtype": tf.float64,
        "name": "mel_matrix_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.linear_to_mel_weight_matrix"] = tf_signal_linear_to_mel_weight_matrix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.linear_to_mel_weight_matrix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.linear_to_mel_weight_matrix'.")

check_valid('tf.signal.linear_to_mel_weight_matrix', generated_inputs['tf.signal.linear_to_mel_weight_matrix'], lib="tf", suffix=0)
