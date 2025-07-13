
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_stop_gradient_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "float_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "int_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    input_tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "negative_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "2d_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "3d_float"}
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
