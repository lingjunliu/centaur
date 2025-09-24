
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    # Input 1: Scalar integer tensor
    tensor = np.int32(5)
    name = "scalar_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float tensor
    tensor = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    name = "1d_float"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer tensor
    tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    name = "2d_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean tensor
    tensor = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_)
    name = "3d_bool"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty 1D tensor
    tensor = np.array([], dtype=np.float64)
    name = "empty_1d"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    tensor = np.array([-1, -2, -3], dtype=np.int32)
    name = "negative_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with zeros
    tensor = np.array([0, 0, 0], dtype=np.int32)
    name = "zero_int"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger tensor
    tensor = np.random.rand(10, 10).astype(np.float32)
    name = "large_float"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String tensor. Use object dtype instead of string_
    tensor = np.array(["hello", "world"], dtype=object)
    name = "string_tensor"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with complex numbers
    tensor = np.array([1+1j, 2+2j], dtype=np.complex64)
    name = "complex_tensor"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_tensor'.")

check_valid('tf.io.serialize_tensor', generated_inputs['tf.io.serialize_tensor'], lib="tf", suffix=0)
