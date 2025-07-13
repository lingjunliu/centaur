
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(4.0, dtype=np.float32)
    name = "sqrt_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    name = "sqrt_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    name = "sqrt_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 3D array
    x = np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]], dtype=np.float32)
    name = "sqrt_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, scalar
    x = np.array(4.0, dtype=np.float64)
    name = "sqrt_scalar_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, scalar
    x = np.array(4.0 + 3.0j, dtype=np.complex64)
    name = "sqrt_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 1D array
    x = np.array([1.0 + 0j, 4.0 + 0j, 0 + 9j], dtype=np.complex128)
    name = "sqrt_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: bfloat16, scalar
    x = np.array(4.0, dtype=np.float32)
    x = tf.dtypes.cast(x, dtype=tf.bfloat16).numpy()
    name = "sqrt_bfloat16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half, scalar
    x = np.array(4.0, dtype=np.float16)
    name = "sqrt_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with zero
    x = np.array(0.0, dtype=np.float32)
    name = "sqrt_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Sqrt"] = tf_raw_ops_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sqrt'.")

check_valid('tf.raw_ops.Sqrt', generated_inputs['tf.raw_ops.Sqrt'], lib="tf", suffix=0)
