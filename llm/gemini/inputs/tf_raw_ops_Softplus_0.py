
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softplus_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float16 tensor
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float16), "name": "softplus_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half tensor
    features = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    input_dict = {"features": features.astype(np.float16), "name": "softplus_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 tensor
    features = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"features": features.astype(np.float64), "name": "softplus_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 tensor
    features = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 tensor
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger float32 values
    features = np.array([-10.0, 10.0, 100.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All negative float32 values
    features = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative float64, different name
    features = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
    input_dict = {"features": features.astype(np.float64), "name": "different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero float32
    features = np.array([0.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "softplus_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_softplus_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softplus'.")

check_valid('tf.raw_ops.Softplus', generated_inputs['tf.raw_ops.Softplus'], lib="tf", suffix=0)
