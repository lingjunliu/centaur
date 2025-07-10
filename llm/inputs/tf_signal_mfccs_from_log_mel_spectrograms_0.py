
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_mfccs_from_log_mel_spectrograms_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.numpy()

    # Input 1: Basic 2D input
    log_mel_spectrograms = np.random.rand(10, 20).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D input
    log_mel_spectrograms = np.random.rand(5, 10, 20).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": "mfcc_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shape
    log_mel_spectrograms = np.random.rand(20, 40).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values
    log_mel_spectrograms = (np.random.rand(10, 20) * 100).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small values
    log_mel_spectrograms = (np.random.rand(10, 20) * 0.01).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 input
    log_mel_spectrograms = np.random.rand(10, 20).astype(np.float64)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D input
    log_mel_spectrograms = np.random.rand(2, 5, 10, 20).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch of one
    log_mel_spectrograms = np.random.rand(1, 20).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different name
    log_mel_spectrograms = np.random.rand(10, 20).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": "different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different name
    log_mel_spectrograms = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"log_mel_spectrograms": tf.constant(log_mel_spectrograms), "name": "another_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
