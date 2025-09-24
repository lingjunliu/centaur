
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_crelu_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default axis
    features = np.array([-1, -2, 3, 4, -5]).astype(np.float32)
    axis = -1
    name = "crelu_basic"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional array
    features = np.array([[-1, 2], [3, -4]]).astype(np.float32)
    axis = -1
    name = "crelu_2d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Axis = 0
    features = np.array([[-1, 2], [3, -4]]).astype(np.float32)
    axis = 0
    name = "crelu_axis0"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Axis = 1
    features = np.array([[-1, 2], [3, -4]]).astype(np.float32)
    axis = 1
    name = "crelu_axis1"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    features = np.random.rand(2, 3, 4).astype(np.float32) - 0.5
    axis = -1
    name = "crelu_3d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (int32)
    features = np.array([-1, 2, -3, 4]).astype(np.int32)
    axis = -1
    name = "crelu_int32"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: 4D array
    features = np.random.rand(2, 3, 4, 5).astype(np.float32) - 0.5
    axis = 2
    name = "crelu_4d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger negative values
    features = np.array([-100, 200, -300, 400]).astype(np.float32)
    axis = -1
    name = "crelu_large_neg"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Only positive values
    features = np.array([1, 2, 3, 4]).astype(np.float32)
    axis = -1
    name = "crelu_pos_only"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    features = np.array([0, 0, 0, 0]).astype(np.float32)
    axis = -1
    name = "crelu_zeros"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.crelu"] = tf_nn_crelu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.crelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.crelu'.")

check_valid('tf.nn.crelu', generated_inputs['tf.nn.crelu'], lib="tf", suffix=0)
