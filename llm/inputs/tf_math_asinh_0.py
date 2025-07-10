
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_asinh_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(1.0)
    name = "asinh_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D tensor
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "asinh_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D tensor
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "asinh_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16, 3D tensor - Removing this as it causes error
    #x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.bfloat16).numpy()
    #name = "asinh_3d"
    #input_dict = {"x": x, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half, scalar
    x = np.float16(0.5)
    name = "asinh_half_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D tensor
    x = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=np.complex64)
    name = "asinh_complex64_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 2D tensor
    x = np.array([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]], dtype=np.complex128)
    name = "asinh_complex128_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, tensor with negative and positive values
    x = np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float32)
    name = "asinh_mixed_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, large values
    x = np.array([100.0, 1000.0, 10000.0], dtype=np.float64)
    name = "asinh_large_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float32, tensor with zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "asinh_zeros"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.asinh"] = tf_math_asinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.asinh'.")

check_valid('tf.math.asinh', generated_inputs['tf.math.asinh'], lib="tf", suffix=0)
