
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_imag_inputs():
    list_of_inputs = []
    
    # Input 1: Complex64 with real and imaginary parts
    input_tensor = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Complex128 with real and imaginary parts
    input_tensor = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Complex64 with negative imaginary parts
    input_tensor = np.array([-1-2j, -3-4j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex128 with mixed real and imaginary parts
    input_tensor = np.array([0+1j, 2+3j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex64 with zero imaginary part
    input_tensor = np.array([1+0j, 3+0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex128 with zero real part
    input_tensor = np.array([0+1j, 0+3j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex64 with large imaginary parts
    input_tensor = np.array([1+100j, 3+200j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex128 with small imaginary parts
    input_tensor = np.array([1+0.1j, 3+0.2j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex64 with complex numbers in 2D array
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex128 with complex numbers in 3D array
    input_tensor = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Imag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Imag', generated_inputs['tf.raw_ops.Imag'], lib="tf", suffix=0)
