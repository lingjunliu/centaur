
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_log_inputs():
    list_of_inputs = []
    
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "log_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.718281828, dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.5, 1.5, 10.0], dtype=np.float16)
    input_dict = {"x": x, "name": "log_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "log_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+0j, 2+0j, 3+0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "log_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.linspace(0.1, 100.0, 50, dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.ones((2, 2, 2, 2), dtype=np.float64)
    input_dict = {"x": x, "name": "log_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Log', generated_inputs['tf.raw_ops.Log'], lib="tf", suffix=0)
