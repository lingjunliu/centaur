
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_1",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1., -2., -3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_2",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([[1., -2., 3.], [-4., 5., -6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_3",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[-1., -2., -3.], [-4., -5., -6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_4",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
    input_dict = {
        "name": "selu_5",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float32)
    input_dict = {
        "name": "selu_6",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_7",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_8",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_9",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_10",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Selu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Selu', generated_inputs['tf.raw_ops.Selu'], lib="tf", suffix=0)
