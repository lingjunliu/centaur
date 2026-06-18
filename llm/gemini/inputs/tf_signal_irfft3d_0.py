
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft3d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_tensor = np.ones((4, 4, 3), dtype=np.complex64)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "irfft3d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2
    input_tensor = np.ones((3, 3, 2), dtype=np.complex128)
    fft_length = np.array([3, 3, 3], dtype=np.int32)
    name = "irfft3d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3
    input_tensor = (np.random.randn(2, 5, 5).astype(np.complex64) + 
                    1j * np.random.randn(2, 5, 5).astype(np.complex64))
    fft_length = np.array([2, 5, 8], dtype=np.int32)
    name = "irfft3d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4
    input_tensor = (np.random.randn(2, 8, 8, 5).astype(np.complex128) + 
                    1j * np.random.randn(2, 8, 8, 5).astype(np.complex128))
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "irfft3d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5
    input_tensor = np.ones((1, 1, 6, 6, 4), dtype=np.complex64)
    fft_length = np.array([6, 6, 6], dtype=np.int32)
    name = "irfft3d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6
    input_tensor = np.zeros((2, 2, 2), dtype=np.complex64)
    fft_length = np.array([2, 2, 2], dtype=np.int32)
    name = "irfft3d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7
    input_tensor = np.ones((5, 5, 10), dtype=np.complex128)
    fft_length = np.array([5, 5, 18], dtype=np.int32)
    name = "irfft3d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8
    input_tensor = np.ones((4, 4, 4, 3), dtype=np.complex64)
    fft_length = np.array([4, 4, 5], dtype=np.int32)
    name = "irfft3d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9
    input_tensor = np.ones((10, 10, 6), dtype=np.complex128)
    fft_length = np.array([10, 10, 10], dtype=np.int32)
    name = "irfft3d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10
    input_tensor = np.ones((8, 12, 7), dtype=np.complex64)
    fft_length = np.array([8, 12, 12], dtype=np.int32)
    name = "irfft3d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.irfft3d"] = tf_signal_irfft3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.irfft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.irfft3d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.irfft3d', generated_inputs['tf.signal.irfft3d'], lib="tf", suffix=0)
