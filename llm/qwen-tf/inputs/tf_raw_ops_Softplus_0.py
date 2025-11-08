
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_softplus_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_1",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_2",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([1.0], dtype=np.float32)
    input_dict = {
        "name": "softplus_3",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_4",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([[-1.0, -2.0, -3.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_5",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "name": "softplus_6",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {
        "name": "softplus_7",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_8",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_9",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_10",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = generate_softplus_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softplus'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Softplus', generated_inputs['tf.raw_ops.Softplus'], lib="tf", suffix=0)
