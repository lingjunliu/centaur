
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, 2D input, exact match fft_length
    input_tensor = np.random.randn(4, 4).astype(np.float32)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2: float32, negative values, 2D input
    input_tensor = np.array([[-1.5, -2.5], [-3.5, -4.5]], dtype=np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "rfft2d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3: float64, 3D input, exact match fft_length
    input_tensor = np.random.randn(2, 3, 3).astype(np.float64)
    fft_length = np.array([3, 3], dtype=np.int32)
    name = "rfft2d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4: float32, 2D input, fft_length is larger (zero-padding)
    input_tensor = np.random.randn(3, 3).astype(np.float32)
    fft_length = np.array([5, 5], dtype=np.int32)
    name = "rfft2d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5: float32, 2D input, fft_length is smaller (cropping)
    input_tensor = np.random.randn(6, 6).astype(np.float32)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6: float64, 4D input, exact match fft_length
    input_tensor = np.random.randn(2, 2, 4, 4).astype(np.float64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7: float32, 2D input, one dimension cropped, one dimension padded
    input_tensor = np.random.randn(4, 8).astype(np.float32)
    fft_length = np.array([6, 6], dtype=np.int32)
    name = "rfft2d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8: Small float32 input
    input_tensor = np.random.randn(2, 2).astype(np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "rfft2d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9: float64, 3D input, mixed cropping and padding
    input_tensor = np.random.randn(1, 10, 10).astype(np.float64)
    fft_length = np.array([5, 12], dtype=np.int32)
    name = "rfft2d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10: float32, 5D input
    input_tensor = np.random.randn(2, 2, 2, 3, 3).astype(np.float32)
    fft_length = np.array([3, 3], dtype=np.int32)
    name = "rfft2d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.rfft2d"] = tf_signal_rfft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.rfft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.rfft2d', generated_inputs['tf.signal.rfft2d'], lib="tf", suffix=0)
