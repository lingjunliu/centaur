
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    features = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    name = "elu_op"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4
    features = np.array([-0.5, 0.5, -1.5, 1.5], dtype=np.float64)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    features = np.array([[-10.0, -5.0], [5.0, 10.0]], dtype=np.float64)
    name = "elu_op_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    features = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    features = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    name = "test_name"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    features = np.array([[-0.001, 0.001], [-0.002, 0.002]], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    features = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10
    features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = None
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_Elu_inputs()
generated_inputs["tf.raw_ops.Elu"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.Elu"].append({"args": (), "kwargs": copy.deepcopy(input_dict)})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
