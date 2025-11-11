
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_imag_inputs():
    list_of_inputs = []

    input1 = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict1 = {'input': input1, 'Tout': np.float32, 'name': 'imag1'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    input_dict2 = {'input': input2, 'Tout': np.float64, 'name': 'imag2'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    input_dict3 = {'input': input3, 'Tout': np.float32, 'name': 'imag3'}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1 - 1j], [0 + 0j]], dtype=np.complex128)
    input_dict4 = {'input': input4, 'Tout': np.float64, 'name': 'imag4'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1j, 2j, 3j], dtype=np.complex64)
    input_dict5 = {'input': input5, 'Tout': np.float32, 'name': 'imag5'}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128)
    input_dict6 = {'input': input6, 'Tout': np.float64, 'name': 'imag6'}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.5 + 2.5j], [3.5 - 4.5j]], dtype=np.complex64)
    input_dict7 = {'input': input7, 'Tout': np.float32, 'name': 'imag7'}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1 + 1j, 2 + 2j]], [[3 + 3j, 4 + 4j]]], dtype=np.complex128)
    input_dict8 = {'input': input8, 'Tout': np.float64, 'name': 'imag8'}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-1 + 1j, -2 + 2j], dtype=np.complex64)
    input_dict9 = {'input': input9, 'Tout': np.float32, 'name': 'imag9'}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.1 + 2.2j, 3.3 + 4.4j, 5.5 + 6.6j], dtype=np.complex128)
    input_dict10 = {'input': input10, 'Tout': np.float64, 'name': 'imag10'}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_raw_ops_imag_inputs()

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
