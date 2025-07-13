
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: bfloat16, 1D
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: half, 2D
    x = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float16)
    name = "sigmoid_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 3D
    x = np.array([[[ -1.0, 0.0], [1.0, 2.0]], [[-2.0, -3.0], [-4.0, 0.0]]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.array(-0.5, dtype=np.float64)
    name = "scalar_sigmoid"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D
    x = np.array([1+1j, 2-2j, 0+0j], dtype=np.complex64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D
    x = np.array([[1+1j, 2-2j], [0+0j, -1-1j]], dtype=np.complex128)
    name = "complex_sigmoid"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, large values
    x = np.array([-100.0, 100.0, 0.0], dtype=np.float16)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, all negative
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float16)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, all positive
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64, array with zero real part
    x = np.array([1j, -1j, 2j, -2j], dtype=np.complex64)
    name = "imaginary_sigmoid"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
sigmoid_inputs = tf_raw_ops_sigmoid_inputs()
generated_inputs["tf.raw_ops.Sigmoid"] = []
for input_dict in sigmoid_inputs:
    x_np = input_dict["x"]
    x = tf.constant(x_np)
    name = input_dict["name"]
    generated_inputs["tf.raw_ops.Sigmoid"].append({"x": x, "name": name})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sigmoid'.")

check_valid('tf.raw_ops.Sigmoid', generated_inputs['tf.raw_ops.Sigmoid'], lib="tf", suffix=0)
