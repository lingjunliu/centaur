
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_tanh_inputs():
    list_of_inputs = []

    # Input 1: float32 tensor
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "tanh_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    name = "tanh_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 tensor
    #x = np.array([-1.0, 0.0, 1.0], dtype=np.float16) #, dtype=tf.bfloat16)
    #name = "tanh_bfloat16"
    #input_dict = {"x": x, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half tensor
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float16) #, dtype=tf.half)
    name = "tanh_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 tensor
    x = np.array([1+1j, -1-1j, 0+0j], dtype=np.complex64)
    name = "tanh_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 tensor
    x = np.array([1+1j, -1-1j, 0+0j, 2-2j], dtype=np.complex128)
    name = "tanh_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional float32 tensor
    x = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    name = "tanh_float32_multidim"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with large values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    name = "tanh_large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with inf and nan
    x = np.array([-np.inf, np.nan, np.inf], dtype=np.float32)
    name = "tanh_inf_nan"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with only zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "tanh_zeros"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.tanh"] = tf_math_tanh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.tanh'.")

check_valid('tf.math.tanh', generated_inputs['tf.math.tanh'], lib="tf", suffix=0)
