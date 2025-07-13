
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []

    # Input 1, valid
    features = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    features = np.array([-5.0, -2.0, 0.0, 3.0], dtype=np.float64)
    name = "elu_op"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, multi-dimensional
    features = np.array([[1, -1], [2, -2]], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, float16
    features = np.array([1.0, -1.0, 0.0], dtype=np.float16)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, half
    features = np.array([1.0, -1.0, 0.0], dtype=np.float16)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, large negative number
    features = np.array([-1000.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, large positive number
    features = np.array([1000.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, zero
    features = np.array([0.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, multi-dimensional array
    features = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, different values
    features = np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float32)
    name = "another_elu"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_Elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
