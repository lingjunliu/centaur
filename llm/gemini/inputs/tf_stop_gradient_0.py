
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_stop_gradient_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "float_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative int32 tensor
    input_tensor = np.array(-5, dtype=np.int32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "negative_int_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "1d_float_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 tensor
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "2d_int_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex64 tensor
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "3d_complex_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bool tensor
    input_tensor = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "bool_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 tensor with negative values
    input_tensor = np.array([[[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]],
                             [[[9.0, -10.0], [11.0, -12.0]], [[-13.0, 14.0], [-15.0, 16.0]]]], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "4d_float_negative_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8 tensor
    input_tensor = np.array([[1, 2], [254, 255]], dtype=np.uint8)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "uint8_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 with zeros
    input_tensor = np.array([[0, 1], [2, 0]], dtype=np.int32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "2d_int_with_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large float 32 tensor
    input_tensor = np.random.rand(100, 100).astype(np.float32) * 1000
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "large_float32_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: tensor with string name
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor).numpy(), "name": "a_string_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.stop_gradient"] = tf_stop_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.stop_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.stop_gradient'.")

check_valid('tf.stop_gradient', generated_inputs['tf.stop_gradient'], lib="tf", suffix=0)
