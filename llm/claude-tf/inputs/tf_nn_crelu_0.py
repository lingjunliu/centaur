
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_crelu_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    axis = -1
    name = "crelu_1d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float64)
    axis = -1
    name = "crelu_2d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    axis = 0
    name = "crelu_3d_axis0"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[10, -20, 30], [-40, 50, -60]], dtype=np.int64)
    axis = 1
    name = "crelu_int64"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1000, -2000], [-3000, 4000]], dtype=np.int16)
    axis = 0
    name = "crelu_int16"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[10, -20], [-30, 40]], dtype=np.int8)
    axis = 1
    name = "crelu_int8"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = -1
    name = "crelu_4d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float32)
    axis = 2
    name = "crelu_3d_axis2"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[5.0, -10.0, 15.0]], dtype=np.float32)
    axis = -1
    name = "crelu_single_row"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
    axis = 0
    name = "crelu_zeros"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.crelu"] = tf_nn_crelu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.crelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.crelu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.crelu', generated_inputs['tf.nn.crelu'], lib="tf", suffix=0)
