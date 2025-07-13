
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_zeros_like_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dtype = np.int32
    name = "zeros_like_int32"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dtype = np.float32
    name = "zeros_like_float32"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shape
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dtype = np.int64
    name = "zeros_like_1d"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool
    input_tensor = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.bool_
    name = "zeros_like_bool"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    dtype = np.complex64
    name = "zeros_like_complex"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimensions
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    dtype = np.float64
    name = "zeros_like_3d"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    input_tensor = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    dtype = np.int32
    name = "zeros_like_negative"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    dtype = np.uint8
    name = "zeros_like_uint8"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16 and specific name
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int16)
    dtype = np.int16
    name = "my_zeros_like_int16"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    dtype = np.float16
    name = "zeros_like_float16"
    layout = None
    input_dict = {"input": input_tensor, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.zeros_like"] = tf_zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.zeros_like'.")

check_valid('tf.zeros_like', generated_inputs['tf.zeros_like'], lib="tf", suffix=0)
