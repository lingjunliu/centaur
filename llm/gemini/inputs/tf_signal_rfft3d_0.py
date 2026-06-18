
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft3d_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 input, exact fft_length
    input_tensor = np.random.randn(4, 4, 4).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2: 4D float32 input (batch dimension), exact fft_length
    input_tensor = np.random.randn(2, 8, 8, 8).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "rfft3d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3: float64 input, exact fft_length
    input_tensor = np.random.randn(3, 5, 7).astype(np.float64)
    fft_length = np.array([3, 5, 7], dtype=np.int32)
    name = "rfft3d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4: High dimensional float32 input
    input_tensor = np.random.randn(1, 2, 2, 4, 4, 4).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5: Crop along all dimensions (fft_length smaller than input)
    input_tensor = np.random.randn(8, 8, 8).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6: Pad along all dimensions (fft_length larger than input)
    input_tensor = np.random.randn(4, 4, 4).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "rfft3d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7: Mixed crop and pad, float64
    input_tensor = np.random.randn(2, 4, 8, 16).astype(np.float64)
    fft_length = np.array([6, 4, 20], dtype=np.int32)
    name = "rfft3d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8: Another valid size mix, float32
    input_tensor = np.random.randn(10, 10, 10).astype(np.float32)
    fft_length = np.array([5, 12, 8], dtype=np.int32)
    name = "rfft3d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9: Large powers of 2 (efficient for FFT)
    input_tensor = np.random.randn(16, 16, 16).astype(np.float32)
    fft_length = np.array([16, 16, 16], dtype=np.int32)
    name = "rfft3d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10: 4D float64 input, exact fft_length
    input_tensor = np.random.randn(2, 2, 2, 2).astype(np.float64)
    fft_length = np.array([2, 2, 2], dtype=np.int32)
    name = "rfft3d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.rfft3d"] = tf_signal_rfft3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.rfft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft3d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.rfft3d', generated_inputs['tf.signal.rfft3d'], lib="tf", suffix=0)
