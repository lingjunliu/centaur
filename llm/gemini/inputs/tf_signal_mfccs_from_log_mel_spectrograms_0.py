
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_mfccs_from_log_mel_spectrograms_inputs():
    list_of_inputs = []

    # Input 1
    log_mel_spectrograms = np.random.rand(10, 80).astype(np.float32)
    name = None
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    log_mel_spectrograms = np.random.rand(5, 128).astype(np.float64)
    name = "mfcc_calculation"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    log_mel_spectrograms = np.random.rand(1, 40).astype(np.float32)
    name = ""
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4
    log_mel_spectrograms = np.random.rand(2, 3, 64).astype(np.float32)
    name = "mfcc_features"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    log_mel_spectrograms = np.random.rand(32, 80).astype(np.float32) - 0.5 # Negative values
    name = None
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    log_mel_spectrograms = np.zeros((10, 60), dtype=np.float32)
    name = "zero_input"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    log_mel_spectrograms = np.random.rand(4, 5, 6, 32).astype(np.float32)
    name = "multi_dimensional"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    log_mel_spectrograms = np.random.rand(16, 128).astype(np.float64) - 1.0
    name = "double_precision"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    log_mel_spectrograms = np.ones((8, 40), dtype=np.float32)
    name = "ones_input"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10
    log_mel_spectrograms = np.random.rand(2, 256).astype(np.float32)
    name = "larger_mel_bins"
    input_dict = {"log_mel_spectrograms": log_mel_spectrograms, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.mfccs_from_log_mel_spectrograms"] = tf_signal_mfccs_from_log_mel_spectrograms_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.mfccs_from_log_mel_spectrograms' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.mfccs_from_log_mel_spectrograms'.")

check_valid('tf.signal.mfccs_from_log_mel_spectrograms', generated_inputs['tf.signal.mfccs_from_log_mel_spectrograms'], lib="tf", suffix=0)
