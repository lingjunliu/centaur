
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_ifft3d_inputs():
    list_of_inputs = []

    # Input 1: Basic complex64
    input_tensor = tf.constant(np.random.rand(2, 4, 8, 16) + 1j * np.random.rand(2, 4, 8, 16), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic complex128
    input_tensor = tf.constant(np.random.rand(2, 4, 8, 16) + 1j * np.random.rand(2, 4, 8, 16), dtype=tf.complex128)
    input_dict = {"input": input_tensor, "name": "my_ifft3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dimensions complex64
    input_tensor = tf.constant(np.random.rand(3, 5, 7, 9) + 1j * np.random.rand(3, 5, 7, 9), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dimensions complex128
    input_tensor = tf.constant(np.random.rand(1, 2, 3, 4) + 1j * np.random.rand(1, 2, 3, 4), dtype=tf.complex128)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger tensor complex64
    input_tensor = tf.constant(np.random.rand(8, 8, 8, 8) + 1j * np.random.rand(8, 8, 8, 8), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with some negative values complex128
    input_tensor = tf.constant(np.random.rand(2, 4, 8, 16) * 2 - 1 + 1j * (np.random.rand(2, 4, 8, 16) * 2 - 1), dtype=tf.complex128)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with only real part complex64
    input_tensor = tf.constant(np.random.rand(2, 4, 8, 16), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with only imaginary part complex128
    input_tensor = tf.constant(1j * np.random.rand(2, 4, 8, 16), dtype=tf.complex128)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small size
    input_tensor = tf.constant(np.random.rand(1, 1, 1, 1) + 1j * np.random.rand(1, 1, 1, 1), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: another different dimensions complex64
    input_tensor = tf.constant(np.random.rand(2, 3, 4, 5) + 1j * np.random.rand(2, 3, 4, 5), dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_signal_ifft3d_inputs()
for i in range(len(inputs)):
    inputs[i]['input'] = inputs[i]['input'].numpy()
generated_inputs["tf.signal.ifft3d"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.ifft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.ifft3d'.")

check_valid('tf.signal.ifft3d', generated_inputs['tf.signal.ifft3d'], lib="tf", suffix=0)
