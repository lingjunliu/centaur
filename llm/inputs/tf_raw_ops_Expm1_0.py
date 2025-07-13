
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_expm1_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([2.0, 8.0, -1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test_expm1_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "test_expm1_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, scalar
    x = np.complex64(1 + 1j)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128, 1D array
    x = np.array([1 + 1j, 2 - 2j], dtype=np.complex128)
    input_dict = {"x": x, "name": "test_expm1_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half, scalar
    x = np.array(3.0, dtype=np.float16)
    input_dict = {"x": x, "name": "test_half_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half, 2D array
    x = np.array([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float16)
    input_dict = {"x": x, "name": "test_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": "test_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, scalar, negative
    x = np.array(-5.0, dtype=np.float64)
    input_dict = {"x": x, "name": "test_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 2D array
    x = np.array([[1+1j, 2-2j], [3+0j, 0-4j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "test_complex_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Expm1"] = tf_raw_ops_expm1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Expm1'.")

check_valid('tf.raw_ops.Expm1', generated_inputs['tf.raw_ops.Expm1'], lib="tf", suffix=0)
