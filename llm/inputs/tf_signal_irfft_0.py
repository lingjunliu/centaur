
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_1"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different fft_length
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    fft_length = np.array([7], dtype=np.int32)
    name = "irfft_2"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different complex type
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_3"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional input
    input_tensor = np.array([[1 + 0j, 2 + 0j, 3 + 0j], [4 + 0j, 5 + 0j, 6 + 0j]], dtype=np.complex64)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_4"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional input, different shape
    input_tensor = np.array([[[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], [[5 + 0j, 6 + 0j], [7 + 0j, 8 + 0j]]], dtype=np.complex64)
    fft_length = np.array([3], dtype=np.int32)
    name = "irfft_5"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: fft_length smaller than input size
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j], dtype=np.complex64)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_6"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger fft_length
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    fft_length = np.array([10], dtype=np.int32)
    name = "irfft_7"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another multi-dimensional input with different shape
    input_tensor = np.array([[[1 + 0j, 2 + 0j, 3 + 0j]]], dtype=np.complex64)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_8"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Different fft length and complex type
    input_tensor = np.array([1 + 0j, 2 + 0j], dtype=np.complex128)
    fft_length = np.array([3], dtype=np.int32)
    name = "irfft_9"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional with different fft_length and shape
    input_tensor = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=np.complex64)
    fft_length = np.array([5], dtype=np.int32)
    name = "irfft_10"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.irfft"] = tf_signal_irfft_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.irfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.irfft'.")

check_valid('tf.signal.irfft', generated_inputs['tf.signal.irfft'], lib="tf", suffix=0)
