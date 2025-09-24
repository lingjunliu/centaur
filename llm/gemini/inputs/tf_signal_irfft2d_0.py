
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1+0j, 2+0j, 3+0j], [4+0j, 5+0j, 6+0j]], dtype=np.complex64)
    fft_length = np.array([2, 4], dtype=np.int32)
    name = "irfft2d_1"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1+0j, 2+0j], [3+0j, 4+0j]], dtype=np.complex128)
    fft_length = np.array([3, 3], dtype=np.int32)
    name = "irfft2d_2"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1+0j, 2+0j], [3+0j, 4+0j]], [[5+0j, 6+0j], [7+0j, 8+0j]]], dtype=np.complex64)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "irfft2d_3"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1+0j, 2+0j, 3+0j], [4+0j, 5+0j, 6+0j], [7+0j, 8+0j, 9+0j]], dtype=np.complex128)
    fft_length = np.array([4, 5], dtype=np.int32)
    name = "irfft2d_4"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "irfft2d_5"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1+0j, 2+0j], [3+0j, 4+0j]], [[5+0j, 6+0j], [7+0j, 8+0j]]], dtype=np.complex128)
    fft_length = np.array([3, 4], dtype=np.int32)
    name = "irfft2d_6"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1+0j, 2+0j, 3+0j, 4+0j], [5+0j, 6+0j, 7+0j, 8+0j]], dtype=np.complex64)
    fft_length = np.array([2, 6], dtype=np.int32)
    name = "irfft2d_7"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1+0j, 2+0j], [3+0j, 4+0j]], dtype=np.complex128)
    fft_length = np.array([4, 2], dtype=np.int32)
    name = "irfft2d_8"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1+0j, 2+0j], [3+0j, 4+0j]], [[5+0j, 6+0j], [7+0j, 8+0j]]], dtype=np.complex64)
    fft_length = np.array([2, 3], dtype=np.int32)
    name = "irfft2d_9"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex128)
    fft_length = np.array([3, 4], dtype=np.int32)
    name = "irfft2d_10"
    input_dict = {"input_tensor": tf.constant(input_tensor).numpy(), "fft_length": tf.constant(fft_length).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.irfft2d"] = tf_signal_irfft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.irfft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.irfft2d'.")

check_valid('tf.signal.irfft2d', generated_inputs['tf.signal.irfft2d'], lib="tf", suffix=0)
