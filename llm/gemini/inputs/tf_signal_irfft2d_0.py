
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft2d_inputs():
    list_of_inputs = []

    # Input 1, valid 2D complex64
    input_tensor = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex64)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2, valid 2D complex128, odd length
    input_tensor = (np.random.randn(6, 3) + 1j * np.random.randn(6, 3)).astype(np.complex128)
    fft_length = np.array([6, 5], dtype=np.int32)
    name = "irfft2d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3, valid 3D complex64
    input_tensor = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(np.complex64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "irfft2d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4, valid 3D complex128, negative values
    input_tensor = (np.random.randn(3, 10, 6) + 1j * np.random.randn(3, 10, 6)).astype(np.complex128)
    fft_length = np.array([10, 11], dtype=np.int32)
    name = "irfft2d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5, valid 4D complex64
    input_tensor = (np.random.randn(1, 2, 8, 5) + 1j * np.random.randn(1, 2, 8, 5)).astype(np.complex64)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6, cropping case
    input_tensor = (np.random.randn(6, 5) + 1j * np.random.randn(6, 5)).astype(np.complex64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "irfft2d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7, padding case
    input_tensor = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex128)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8, odd dimensions
    input_tensor = (np.random.randn(7, 4) + 1j * np.random.randn(7, 4)).astype(np.complex64)
    fft_length = np.array([7, 7], dtype=np.int32)
    name = "irfft2d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9, larger size
    input_tensor = (np.random.randn(32, 17) + 1j * np.random.randn(32, 17)).astype(np.complex64)
    fft_length = np.array([32, 32], dtype=np.int32)
    name = "irfft2d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10, multibatch with mixed dimensions
    input_tensor = (np.random.randn(2, 2, 5, 4) + 1j * np.random.randn(2, 2, 5, 4)).astype(np.complex128)
    fft_length = np.array([5, 6], dtype=np.int32)
    name = "irfft2d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.irfft2d"] = tf_signal_irfft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.irfft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.irfft2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.irfft2d', generated_inputs['tf.signal.irfft2d'], lib="tf", suffix=0)
