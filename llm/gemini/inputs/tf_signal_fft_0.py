
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft_inputs():
    list_of_inputs = []

    # Input 1: Basic complex64 array
    input1 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict1 = {"input": input1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic complex128 array
    input2 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    input_dict2 = {"input": input2, "name": "fft_input2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional complex64 array
    input3 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict3 = {"input": input3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional complex128 array
    input4 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict4 = {"input": input4, "name": "fft_input4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex64 array with negative values
    input5 = np.array([-1-1j, -2-2j, -3-3j], dtype=np.complex64)
    input_dict5 = {"input": input5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex128 array with negative values
    input6 = np.array([-1-1j, -2-2j, -3-3j], dtype=np.complex128)
    input_dict6 = {"input": input6, "name": "fft_input6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Complex64 array with zero values
    input7 = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    input_dict7 = {"input": input7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Complex128 array with zero values
    input8 = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex128)
    input_dict8 = {"input": input8, "name": "fft_input8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Complex64 array with different real and imaginary parts
    input9 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict9 = {"input": input9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Complex128 array with different real and imaginary parts
    input10 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
    input_dict10 = {"input": input10, "name": "fft_input10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fft"] = tf_signal_fft_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fft'.")

check_valid('tf.signal.fft', generated_inputs['tf.signal.fft'], lib="tf", suffix=0)
