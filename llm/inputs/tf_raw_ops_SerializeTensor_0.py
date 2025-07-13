
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_serialize_tensor_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    tensor = tf.constant(1)
    name = None
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor of integers
    tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    name = "my_tensor"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor of floats
    tensor = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    name = ""
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor of booleans
    tensor = tf.constant(np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=np.bool_))
    name = "bool_tensor"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor of strings
    tensor = tf.constant(["hello", "world"])
    name = None
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SerializeTensor"] = tf_raw_ops_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SerializeTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SerializeTensor'.")

check_valid('tf.raw_ops.SerializeTensor', generated_inputs['tf.raw_ops.SerializeTensor'], lib="tf", suffix=0)
