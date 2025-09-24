
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    fft_length = np.array([4], dtype=np.int32)
    name = "rfft_1"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    fft_length = np.array([5], dtype=np.int32)
    name = "rfft_2"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    fft_length = np.array([3], dtype=np.int32)
    name = "rfft_3"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    fft_length = np.array([3], dtype=np.int32)
    name = "rfft_4"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fft_length = np.array([8], dtype=np.int32)
    name = "rfft_5"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    fft_length = np.array([4], dtype=np.int32)
    name = "rfft_6"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    fft_length = np.array([2], dtype=np.int32)
    name = "rfft_7"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0], dtype=np.float64)
    fft_length = np.array([4], dtype=np.int32)
    name = "rfft_8"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    fft_length = np.array([8], dtype=np.int32)
    name = "rfft_9"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1.0], dtype=np.float64)
    fft_length = np.array([1], dtype=np.int32)
    name = "rfft_10"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.rfft"] = tf_signal_rfft_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.rfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft'.")

check_valid('tf.signal.rfft', generated_inputs['tf.signal.rfft'], lib="tf", suffix=0)
