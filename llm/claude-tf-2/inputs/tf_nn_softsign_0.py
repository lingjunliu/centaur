
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_op4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.001, -0.001, 0.0001], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([100.0, -100.0, 1000.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"features": features, "name": "softsign_op8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array(5.0, dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.5, -2.5, 3.5], [4.5, -5.5, 6.5]], dtype=np.float16)
    input_dict = {"features": features, "name": "softsign_op10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(100, 50).astype(np.float32)
    input_dict = {"features": features, "name": "softsign_op11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softsign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.softsign', generated_inputs['tf.nn.softsign'], lib="tf", suffix=0)
