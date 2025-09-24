
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.complex64(np.random.rand(8, 8, 8, 5))
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "irfft3d_1"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.complex128(np.random.rand(4, 4, 4, 3))
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "irfft3d_2"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - different fft_length
    input_tensor = np.complex64(np.random.rand(8, 8, 8, 5))
    fft_length = np.array([16, 16, 16], dtype=np.int32)
    name = "irfft3d_3"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - smaller fft_length
    input_tensor = np.complex128(np.random.rand(16, 16, 16, 9))
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "irfft3d_4"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5 - rectangular input shape
    input_tensor = np.complex64(np.random.rand(4, 8, 16, 9))
    fft_length = np.array([4, 8, 16], dtype=np.int32)
    name = "irfft3d_5"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 -  complex128
    input_tensor = np.complex128(np.random.rand(2, 2, 2, 2))
    fft_length = np.array([2, 2, 2], dtype=np.int32)
    name = "irfft3d_6"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 -  complex64
    input_tensor = np.complex64(np.random.rand(3, 3, 3, 2))
    fft_length = np.array([3, 3, 3], dtype=np.int32)
    name = "irfft3d_7"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - different dimensions
    input_tensor = np.complex64(np.random.rand(5, 6, 7, 4))
    fft_length = np.array([5, 6, 7], dtype=np.int32)
    name = "irfft3d_8"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - complex128, different dimensions
    input_tensor = np.complex128(np.random.rand(3, 5, 7, 5))
    fft_length = np.array([3, 5, 7], dtype=np.int32)
    name = "irfft3d_9"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - varying fft lengths
    input_tensor = np.complex64(np.random.rand(4, 5, 6, 4))
    fft_length = np.array([8, 10, 12], dtype=np.int32)
    name = "irfft3d_10"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.irfft3d"] = tf_signal_irfft3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.irfft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.irfft3d'.")

check_valid('tf.signal.irfft3d', generated_inputs['tf.signal.irfft3d'], lib="tf", suffix=0)
