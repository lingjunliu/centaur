
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ZerosLike_inputs():
    list_of_inputs = []

    # Input 1: int32, 1D
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 2D
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "float_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, 3D
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 1D, negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "negative_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, 2D
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "complex_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bool, 2D
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int16, 4D
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int16)
    input_dict = {"x": x, "name": "4d_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16, 1D
    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: string name
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x, "name": "my_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_ZerosLike_inputs()
generated_inputs["tf.raw_ops.ZerosLike"] = []
for input_dict in inputs:
    x_np = input_dict["x"]
    x = tf.constant(x_np)
    generated_inputs["tf.raw_ops.ZerosLike"].append({"x": x, "name": input_dict["name"]})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ZerosLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ZerosLike'.")

check_valid('tf.raw_ops.ZerosLike', generated_inputs['tf.raw_ops.ZerosLike'], lib="tf", suffix=0)
