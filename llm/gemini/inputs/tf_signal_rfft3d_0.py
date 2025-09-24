
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft3d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_tensor = np.random.rand(8, 8, 8).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "fft_1"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dimensions
    input_tensor = np.random.rand(16, 16, 16).astype(np.float32)
    fft_length = np.array([16, 16, 16], dtype=np.int32)
    name = "fft_2"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Cropping
    input_tensor = np.random.rand(16, 16, 16).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "fft_3"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Padding
    input_tensor = np.random.rand(8, 8, 8).astype(np.float32)
    fft_length = np.array([16, 16, 16], dtype=np.int32)
    name = "fft_4"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (float64)
    input_tensor = np.random.rand(8, 8, 8).astype(np.float64)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "fft_5"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-cube input
    input_tensor = np.random.rand(8, 16, 32).astype(np.float32)
    fft_length = np.array([8, 16, 32], dtype=np.int32)
    name = "fft_6"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Cropping with different lengths
    input_tensor = np.random.rand(16, 16, 16).astype(np.float32)
    fft_length = np.array([8, 12, 10], dtype=np.int32)
    name = "fft_7"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Padding with different lengths
    input_tensor = np.random.rand(8, 8, 8).astype(np.float32)
    fft_length = np.array([16, 20, 24], dtype=np.int32)
    name = "fft_8"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Smaller input size
    input_tensor = np.random.rand(4, 4, 4).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "fft_9"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different input size and fft_length
    input_tensor = np.random.rand(5, 6, 7).astype(np.float32)
    fft_length = np.array([5, 6, 7], dtype=np.int32)
    name = "fft_10"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.rfft3d"] = tf_signal_rfft3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.rfft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft3d'.")

check_valid('tf.signal.rfft3d', generated_inputs['tf.signal.rfft3d'], lib="tf", suffix=0)
